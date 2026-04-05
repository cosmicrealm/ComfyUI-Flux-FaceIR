from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import math

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from einops import rearrange

from .utils import (
    build_grouped_image_condition_ids,
    build_image_condition_ids,
    get_schedule,
    pack_latents,
    pack_text,
    unpack_latents,
)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(device: torch.device, mixed_precision: str) -> torch.dtype:
    if device.type != "cuda" or mixed_precision == "no":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    return torch.bfloat16


def maybe_autocast(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def preprocess_image(path: str | Path, resolution: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image = Image.open(path).convert("RGB")
    return transform(image)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.clamp(-1, 1)
    image = ((image + 1.0) * 127.5).permute(1, 2, 0).cpu().to(torch.uint8).numpy()
    return Image.fromarray(image)


def _get_comfy_model_management():
    import comfy.model_management

    return comfy.model_management


def _prepare_comfy_image_batch(images: torch.Tensor | None, resolution: int) -> torch.Tensor | None:
    if images is None:
        return None
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(f"Expected image batch with 3 or 4 dims, got shape {tuple(images.shape)}")
    images = images.to(dtype=torch.float32).clamp(0.0, 1.0)
    if resolution <= 0:
        return images

    x = images.permute(0, 3, 1, 2)
    h, w = x.shape[-2:]
    if h != resolution or w != resolution:
        scale = resolution / float(min(h, w))
        new_h = max(resolution, int(round(h * scale)))
        new_w = max(resolution, int(round(w * scale)))
        x = torch.nn.functional.interpolate(
            x,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
        )
        top = max(0, (new_h - resolution) // 2)
        left = max(0, (new_w - resolution) // 2)
        x = x[:, :, top : top + resolution, left : left + resolution]
    return x.permute(0, 2, 3, 1).contiguous()


def _encode_prompt_with_clip(clip, prompt: str, *, guidance_distilled: bool):
    prompt_tokens = clip.tokenize(prompt)
    prompt_ctx, prompt_pooled = clip.encode_from_tokens(prompt_tokens, return_pooled=True)

    if guidance_distilled:
        return prompt_ctx, prompt_pooled, None, None

    empty_tokens = clip.tokenize("")
    empty_ctx, empty_pooled = clip.encode_from_tokens(empty_tokens, return_pooled=True)
    return prompt_ctx, prompt_pooled, empty_ctx, empty_pooled


def _build_txt_ids(diffusion_model, batch_size: int, text_length: int, device: torch.device) -> torch.Tensor:
    dims = len(diffusion_model.params.axes_dim)
    txt_ids = torch.zeros((batch_size, text_length, dims), device=device, dtype=torch.float32)
    txt_id_dims = getattr(diffusion_model.params, "txt_ids_dims", [])
    if len(txt_id_dims) > 0:
        positions = torch.linspace(0, text_length - 1, steps=text_length, device=device, dtype=torch.float32)
        for dim_index in txt_id_dims:
            txt_ids[:, :, dim_index] = positions
    return txt_ids


def _build_condition_tokens_comfy(
    diffusion_model,
    *,
    batch_size: int,
    condition_latents: torch.Tensor | None,
    reference_latents: torch.Tensor | None,
    transformer_options: dict,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    cond_seq = None
    cond_ids = None

    if condition_latents is not None:
        cond_latents = condition_latents
        if cond_latents.shape[0] == 1 and batch_size > 1:
            cond_latents = cond_latents.repeat(batch_size, 1, 1, 1)
        seq, ids = diffusion_model.process_img(cond_latents, index=2, transformer_options=transformer_options)
        cond_seq = seq
        cond_ids = ids

    if reference_latents is not None:
        for ref_index in range(reference_latents.shape[0]):
            ref_latent = reference_latents[ref_index : ref_index + 1]
            if batch_size > 1:
                ref_latent = ref_latent.repeat(batch_size, 1, 1, 1)
            seq, ids = diffusion_model.process_img(
                ref_latent,
                index=10 + ref_index,
                transformer_options=transformer_options,
            )
            if cond_seq is None:
                cond_seq = seq
                cond_ids = ids
            else:
                cond_seq = torch.cat((cond_seq, seq), dim=1)
                cond_ids = torch.cat((cond_ids, ids), dim=1)

    return cond_seq, cond_ids


@torch.no_grad()
def sample_image_comfy(
    model,
    clip,
    vae,
    prompt: str,
    resolution: int,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    condition_image: torch.Tensor | None = None,
    reference_image: torch.Tensor | None = None,
    lora_path: str | None = None,
    progress_callback=None,
) -> Image.Image:
    comfy_model_management = _get_comfy_model_management()
    comfy_model_management.load_models_gpu([model], force_full_load=True)

    base_model = model.model
    diffusion_model = model.get_model_object("diffusion_model")
    latent_format = model.get_model_object("latent_format")

    device = model.load_device
    dtype = base_model.get_dtype_inference() if hasattr(base_model, "get_dtype_inference") else diffusion_model.dtype
    guidance_distilled = bool(getattr(diffusion_model.params, "guidance_embed", False))

    from .lora import temporary_lora

    latent_h = resolution // 16
    latent_w = resolution // 16
    latent_channels = latent_format.latent_channels
    patch_size = getattr(diffusion_model, "patch_size", 1)
    seq_len = ((latent_h + (patch_size // 2)) // patch_size) * ((latent_w + (patch_size // 2)) // patch_size)

    face_batch = _prepare_comfy_image_batch(condition_image, resolution)
    ref_batch = _prepare_comfy_image_batch(reference_image, resolution)

    condition_latents = vae.encode(face_batch) if face_batch is not None else None
    reference_latents = vae.encode(ref_batch) if ref_batch is not None else None

    if condition_latents is not None:
        condition_latents = condition_latents.to(device=device, dtype=dtype)
    if reference_latents is not None:
        reference_latents = reference_latents.to(device=device, dtype=dtype)

    prompt_ctx, prompt_pooled, empty_ctx, empty_pooled = _encode_prompt_with_clip(
        clip,
        prompt,
        guidance_distilled=guidance_distilled,
    )

    if guidance_distilled:
        ctx = prompt_ctx.to(device=device, dtype=dtype)
        pooled = None if prompt_pooled is None else prompt_pooled.to(device=device, dtype=dtype)
    else:
        ctx = torch.cat((empty_ctx, prompt_ctx), dim=0).to(device=device, dtype=dtype)
        if empty_pooled is not None and prompt_pooled is not None:
            pooled = torch.cat((empty_pooled, prompt_pooled), dim=0).to(device=device, dtype=dtype)
        else:
            pooled = None

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn((1, latent_channels, latent_h, latent_w), device=device, dtype=dtype, generator=generator)
    schedule = get_schedule(num_steps, seq_len)

    transformer_options: dict = {}

    with temporary_lora(diffusion_model, lora_path, target_patterns="flux2_klein"):
        total_steps = len(schedule) - 1
        for step_index, (t_curr, t_next) in enumerate(zip(schedule[:-1], schedule[1:]), start=1):
            dt = t_next - t_curr

            if guidance_distilled:
                batch_latents = latents
                timestep = torch.full((1,), t_curr, device=device, dtype=dtype)
                batch_ctx = ctx
                batch_pooled = pooled
                guidance = torch.full((1,), guidance_scale, device=device, dtype=dtype)
            else:
                batch_latents = torch.cat((latents, latents), dim=0)
                timestep = torch.full((2,), t_curr, device=device, dtype=dtype)
                batch_ctx = ctx
                batch_pooled = pooled
                guidance = None

            img, img_ids = diffusion_model.process_img(batch_latents, transformer_options=transformer_options)
            noise_tokens = img.shape[1]
            cond_seq, cond_ids = _build_condition_tokens_comfy(
                diffusion_model,
                batch_size=batch_latents.shape[0],
                condition_latents=condition_latents,
                reference_latents=reference_latents,
                transformer_options=transformer_options,
            )
            if cond_seq is not None:
                img = torch.cat((img, cond_seq), dim=1)
                img_ids = torch.cat((img_ids, cond_ids), dim=1)

            txt_ids = _build_txt_ids(diffusion_model, batch_latents.shape[0], batch_ctx.shape[1], device=device)
            pred = diffusion_model.forward_orig(
                img,
                img_ids,
                batch_ctx,
                txt_ids,
                timestep,
                batch_pooled,
                guidance,
                None,
                transformer_options=transformer_options,
                attn_mask=None,
            )
            pred = pred[:, :noise_tokens]
            h_len = ((latent_h + (patch_size // 2)) // patch_size)
            w_len = ((latent_w + (patch_size // 2)) // patch_size)
            pred = rearrange(
                pred,
                "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                h=h_len,
                w=w_len,
                ph=patch_size,
                pw=patch_size,
            )[:, :, :latent_h, :latent_w]

            if guidance_distilled:
                latents = latents + dt * pred
            else:
                pred_uncond, pred_cond = pred.chunk(2, dim=0)
                latents = latents + dt * (pred_uncond + guidance_scale * (pred_cond - pred_uncond))

            if progress_callback is not None:
                progress_callback(step_index, total_steps)

    decoded = vae.decode(latents)[0].movedim(-1, 0).to(dtype=torch.float32)
    return tensor_to_pil(decoded * 2.0 - 1.0)


def _encode_condition_tokens(
    ae,
    *,
    condition_image: torch.Tensor | None,
    reference_image: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
    mixed_precision: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    cond_seq = None
    cond_ids = None

    if condition_image is not None:
        if condition_image.ndim == 3:
            condition_image = condition_image.unsqueeze(0)
        with maybe_autocast(device, mixed_precision):
            condition_latents = ae.encode(condition_image.to(device=device, dtype=dtype))
        cond_seq, cond_ids = build_image_condition_ids(condition_latents, time_offset=2)
        cond_seq = cond_seq.to(device=device, dtype=dtype)
        cond_ids = cond_ids.to(device=device)

    if reference_image is not None:
        if reference_image.ndim == 3:
            reference_image = reference_image.unsqueeze(0)
        with maybe_autocast(device, mixed_precision):
            reference_latents = ae.encode(reference_image.to(device=device, dtype=dtype))
        reference_latents = reference_latents.unsqueeze(0)
        ref_seq, ref_ids, _ = build_grouped_image_condition_ids(reference_latents, time_offset=10, seq_offset=0)
        ref_seq = ref_seq.to(device=device, dtype=dtype)
        ref_ids = ref_ids.to(device=device)
        if cond_seq is None:
            cond_seq = ref_seq
            cond_ids = ref_ids
        else:
            cond_seq = torch.cat((cond_seq, ref_seq), dim=1)
            cond_ids = torch.cat((cond_ids, ref_ids), dim=1)

    return cond_seq, cond_ids


@torch.no_grad()
def sample_image(
    transformer,
    ae,
    text_encoder,
    model_cfg: dict,
    prompt: str,
    resolution: int,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    mixed_precision: str,
    condition_image: torch.Tensor | None = None,
    reference_image: torch.Tensor | None = None,
    use_guidance_embed: bool | None = None,
    progress_callback=None,
) -> Image.Image:
    latent_h = resolution // 16
    latent_w = resolution // 16
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn((1, 128, latent_h, latent_w), device=device, dtype=dtype, generator=generator)
    packed_latents, img_ids = pack_latents(latents)
    schedule = get_schedule(num_steps, packed_latents.shape[1])

    cond_seq, cond_ids = _encode_condition_tokens(
        ae,
        condition_image=condition_image,
        reference_image=reference_image,
        device=device,
        dtype=dtype,
        mixed_precision=mixed_precision,
    )

    if use_guidance_embed is None:
        use_guidance_embed = getattr(transformer, "use_guidance_embed", False)

    if model_cfg.get("guidance_distilled", False):
        ctx = text_encoder([prompt]).to(device=device, dtype=dtype)
        ctx, ctx_ids = pack_text(ctx)
        guidance = torch.full((1,), guidance_scale, device=device, dtype=dtype) if use_guidance_embed else None

        total_steps = len(schedule) - 1
        for step_index, (t_curr, t_next) in enumerate(zip(schedule[:-1], schedule[1:]), start=1):
            dt = t_next - t_curr
            timestep = torch.full((1,), t_curr, device=device, dtype=dtype)
            x_in = packed_latents
            ids_in = img_ids
            if cond_seq is not None:
                x_in = torch.cat((x_in, cond_seq), dim=1)
                ids_in = torch.cat((ids_in, cond_ids), dim=1)
            with maybe_autocast(device, mixed_precision):
                pred = transformer(
                    x=x_in,
                    x_ids=ids_in,
                    timesteps=timestep,
                    ctx=ctx,
                    ctx_ids=ctx_ids,
                    guidance=guidance,
                )
            pred = pred[:, : packed_latents.shape[1]]
            packed_latents = packed_latents + dt * pred
            if progress_callback is not None:
                progress_callback(step_index, total_steps)
    else:
        ctx = text_encoder(["", prompt]).to(device=device, dtype=dtype)
        ctx, ctx_ids = pack_text(ctx)

        total_steps = len(schedule) - 1
        for step_index, (t_curr, t_next) in enumerate(zip(schedule[:-1], schedule[1:]), start=1):
            dt = t_next - t_curr
            timestep = torch.full((2,), t_curr, device=device, dtype=dtype)
            x_in = torch.cat((packed_latents, packed_latents), dim=0)
            ids_in = torch.cat((img_ids, img_ids), dim=0)
            if cond_seq is not None:
                cond_seq_batch = torch.cat((cond_seq, cond_seq), dim=0)
                cond_ids_batch = torch.cat((cond_ids, cond_ids), dim=0)
                x_in = torch.cat((x_in, cond_seq_batch), dim=1)
                ids_in = torch.cat((ids_in, cond_ids_batch), dim=1)
            with maybe_autocast(device, mixed_precision):
                pred = transformer(
                    x=x_in,
                    x_ids=ids_in,
                    timesteps=timestep,
                    ctx=ctx,
                    ctx_ids=ctx_ids,
                    guidance=None,
                )
            pred = pred[:, : packed_latents.shape[1]]
            pred_uncond, pred_cond = pred.chunk(2, dim=0)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            packed_latents = packed_latents + dt * pred
            if progress_callback is not None:
                progress_callback(step_index, total_steps)

    decoded_latents = unpack_latents(packed_latents, img_ids)
    with maybe_autocast(device, mixed_precision):
        image = ae.decode(decoded_latents)[0].float()
    return tensor_to_pil(image)


def resize_for_panel(
    image: Image.Image,
    resolution: int | None = None,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    image = image.convert("RGB")
    if target_size is not None:
        if image.size == target_size:
            return image
        return image.resize(target_size, Image.Resampling.BICUBIC)
    if resolution is None or resolution <= 0:
        return image
    return image.resize((resolution, resolution), Image.Resampling.BICUBIC)


def make_comparison_row(
    condition_images: list[tuple[str, Image.Image]] | None,
    reference_images: list[Image.Image],
    target_image: Image.Image | None,
    variant_images: list[tuple[str, Image.Image]],
    resolution: int | None,
    target_before_references: bool = False,
) -> Image.Image:
    reference_tile: Image.Image | None = None
    if condition_images:
        reference_tile = condition_images[0][1]
    elif reference_images:
        reference_tile = reference_images[0]
    elif target_image is not None:
        reference_tile = target_image
    elif variant_images:
        reference_tile = variant_images[0][1]

    if reference_tile is None:
        raise ValueError("At least one image is required to build a comparison row.")

    target_size = None if resolution is not None and resolution > 0 else reference_tile.convert("RGB").size
    tile_width = target_size[0] if target_size is not None else resolution
    tile_height = target_size[1] if target_size is not None else resolution

    labels = []
    tiles = []

    if condition_images:
        for label, image in condition_images:
            labels.append(label)
            tiles.append(resize_for_panel(image, resolution=resolution, target_size=target_size))

    if target_image is not None and target_before_references:
        labels.append("target")
        tiles.append(resize_for_panel(target_image, resolution=resolution, target_size=target_size))

    labels.extend(f"ref{i + 1}" for i in range(len(reference_images)))
    tiles.extend(resize_for_panel(image, resolution=resolution, target_size=target_size) for image in reference_images)

    if target_image is not None and not target_before_references:
        labels.append("target")
        tiles.append(resize_for_panel(target_image, resolution=resolution, target_size=target_size))

    for label, image in variant_images:
        labels.append(label)
        tiles.append(resize_for_panel(image, resolution=resolution, target_size=target_size))

    max_label_lines = max((label.count("\n") + 1 for label in labels), default=1)
    label_height = 8 + max_label_lines * 14
    canvas = Image.new("RGB", (tile_width * len(tiles), tile_height + label_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (label, tile) in enumerate(zip(labels, tiles)):
        x = index * tile_width
        canvas.paste(tile, (x, label_height))
        draw.rectangle((x, 0, x + tile_width, label_height), fill=(240, 240, 240))
        draw.multiline_text((x + 6, 4), label, fill=(0, 0, 0), font=font, spacing=2)

    return canvas


def stack_rows(rows: list[Image.Image]) -> Image.Image:
    if not rows:
        raise ValueError("rows must not be empty")

    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    return canvas
