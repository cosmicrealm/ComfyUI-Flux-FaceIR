from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

DEFAULT_MODEL_NAME = "flux.2-klein-base-4b"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LORA_PATH = "pretrained_models/lora_weights.safetensors"
DEFAULT_OUTPUT_DIR = "outputs/release_lora_ref"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CASE_MODES = ("blind", "ref-single", "ref-multi")


def _get_model_utils():
    from .utils import MODEL_CONFIGS, load_ae, load_transformer

    return MODEL_CONFIGS, load_ae, load_transformer


def _get_lora_ops():
    from .lora import (
        get_target_patterns,
        inject_lora,
        load_lora_checkpoint,
        load_lora_state_dict,
        lora_state_dict_from_full_state_dict,
    )

    return (
        get_target_patterns,
        inject_lora,
        load_lora_checkpoint,
        load_lora_state_dict,
        lora_state_dict_from_full_state_dict,
    )


def _get_inference_ops():
    from .inference import (
        make_comparison_row,
        preprocess_image,
        resolve_device,
        resolve_dtype,
        sample_image,
    )

    return make_comparison_row, preprocess_image, resolve_device, resolve_dtype, sample_image


def _get_text_encoder_loader():
    from .text_encoder import load_text_encoder

    return load_text_encoder


def resolve_release_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str((PACKAGE_ROOT / path).resolve())


def resolve_model_paths(
    *,
    model_name: str,
    model_dir: str | None,
    transformer_path: str | None,
    ae_path: str | None,
    text_encoder_path: str | None,
) -> tuple[str | None, str | None, str | None]:
    MODEL_CONFIGS, _, _ = _get_model_utils()
    model_cfg = MODEL_CONFIGS[model_name]
    transformer_path = transformer_path.strip() if transformer_path else None
    ae_path = ae_path.strip() if ae_path else None
    text_encoder_path = text_encoder_path.strip() if text_encoder_path else None

    if model_dir:
        root = Path(model_dir)
        if transformer_path is None and model_cfg.get("transformer_file"):
            candidate = root / model_cfg["transformer_file"]
            if candidate.exists():
                transformer_path = str(candidate)
        if ae_path is None and model_cfg.get("ae_subdir"):
            candidate = root / model_cfg["ae_subdir"]
            if candidate.exists():
                ae_path = str(candidate)
        if text_encoder_path is None and model_cfg.get("text_encoder_subdir"):
            candidate = root / model_cfg["text_encoder_subdir"]
            if candidate.exists():
                text_encoder_path = str(candidate)

    return transformer_path, ae_path, text_encoder_path


def _load_transformer_with_lora(
    *,
    model_name: str,
    model_dir: str | None,
    transformer_path: str | None,
    lora_path: str | None,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_targets: str,
    device: torch.device,
    dtype: torch.dtype,
):
    _, _, load_transformer = _get_model_utils()
    (
        get_target_patterns,
        inject_lora,
        load_lora_checkpoint,
        load_lora_state_dict,
        lora_state_dict_from_full_state_dict,
    ) = _get_lora_ops()
    transformer_path, _, _ = resolve_model_paths(
        model_name=model_name,
        model_dir=model_dir,
        transformer_path=transformer_path,
        ae_path=None,
        text_encoder_path=None,
    )
    transformer = load_transformer(model_name, weight_path=transformer_path, device=device).to(dtype=dtype)

    if lora_path:
        adapter_state, metadata = load_lora_checkpoint(lora_path)
        preset = metadata.get("lora_target_preset", lora_targets)
        rank = int(metadata.get("lora_rank", lora_rank))
        alpha = float(metadata.get("lora_alpha", lora_alpha))
        dropout = float(metadata.get("lora_dropout", lora_dropout))
        inject_lora(transformer, get_target_patterns(preset), rank=rank, alpha=alpha, dropout=dropout)
        load_lora_state_dict(transformer, lora_state_dict_from_full_state_dict(adapter_state), strict=True)

    transformer.eval()
    transformer.requires_grad_(False)
    return transformer


def _collect_reference_tensors(reference_paths: list[Path], resolution: int) -> torch.Tensor | None:
    if not reference_paths:
        return None
    _, preprocess_image, _, _, _ = _get_inference_ops()
    return torch.stack([preprocess_image(path, resolution) for path in reference_paths])


def create_pipeline(
    *,
    model: str = DEFAULT_MODEL_NAME,
    model_dir: str | None = None,
    transformer_path: str | None = None,
    ae_path: str | None = None,
    text_encoder_path: str | None = None,
    lora_path: str | None = DEFAULT_LORA_PATH,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_targets: str = "flux2_klein",
    device: str | None = None,
    mixed_precision: str = "bf16",
):
    MODEL_CONFIGS, load_ae, _ = _get_model_utils()
    _, _, resolve_device, resolve_dtype, _ = _get_inference_ops()
    load_text_encoder = _get_text_encoder_loader()
    model_name = model.lower()
    model_cfg = MODEL_CONFIGS[model_name]
    resolved_device = resolve_device(device)
    dtype = resolve_dtype(resolved_device, mixed_precision)
    resolved_lora_path = resolve_release_path(lora_path) if lora_path else None

    transformer = _load_transformer_with_lora(
        model_name=model_name,
        model_dir=model_dir,
        transformer_path=transformer_path,
        lora_path=resolved_lora_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
        device=resolved_device,
        dtype=dtype,
    )

    _, ae_path, text_encoder_path = resolve_model_paths(
        model_name=model_name,
        model_dir=model_dir,
        transformer_path=None,
        ae_path=ae_path,
        text_encoder_path=text_encoder_path,
    )

    ae = load_ae(model_name, weight_path=ae_path, device=resolved_device).to(dtype=dtype)
    ae.eval()
    ae.requires_grad_(False)

    text_encoder = load_text_encoder(
        variant=model_cfg["text_encoder_variant"],
        device=resolved_device,
        model_path=text_encoder_path,
        model_dir=model_dir,
    ).to(dtype=dtype)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    defaults = model_cfg.get("defaults", {})
    use_guidance_embed = getattr(transformer, "use_guidance_embed", False)

    return {
        "model_name": model_name,
        "model_cfg": model_cfg,
        "device": resolved_device,
        "dtype": dtype,
        "mixed_precision": mixed_precision,
        "transformer": transformer,
        "ae": ae,
        "text_encoder": text_encoder,
        "default_num_steps": int(defaults.get("num_steps", 50)),
        "default_guidance_scale": float(defaults.get("guidance", 4.0)),
        "use_guidance_embed": use_guidance_embed,
    }


def run_inference(
    *,
    runtime: dict,
    prompt: str,
    resolution: int,
    condition_image_path: str | Path | None = None,
    reference_image_paths: list[str | Path] | None = None,
    num_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int = 42,
) -> Image.Image:
    _, preprocess_image, _, _, sample_image = _get_inference_ops()
    device = runtime["device"]
    dtype = runtime["dtype"]
    mixed_precision = runtime["mixed_precision"]
    model_cfg = runtime["model_cfg"]

    condition_tensor = (
        preprocess_image(condition_image_path, resolution)
        if condition_image_path is not None
        else None
    )
    reference_paths = [Path(path) for path in (reference_image_paths or [])]
    reference_tensor = _collect_reference_tensors(reference_paths, resolution)

    if num_steps is None:
        num_steps = runtime["default_num_steps"]
    if guidance_scale is None:
        guidance_scale = runtime["default_guidance_scale"]

    return sample_image(
        transformer=runtime["transformer"],
        ae=runtime["ae"],
        text_encoder=runtime["text_encoder"],
        model_cfg=model_cfg,
        prompt=prompt,
        resolution=resolution,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        device=device,
        dtype=dtype,
        mixed_precision=mixed_precision,
        condition_image=condition_tensor,
        reference_image=reference_tensor,
        use_guidance_embed=runtime["use_guidance_embed"],
    )


def build_comparison_row(
    *,
    resolution: int,
    output_image: Image.Image,
    condition_image_path: str | Path | None = None,
    reference_image_paths: list[str | Path] | None = None,
    target_image_path: str | Path | None = None,
    output_label: str = "output",
) -> Image.Image:
    make_comparison_row, _, _, _, _ = _get_inference_ops()
    condition_images = None
    if condition_image_path is not None:
        degraded = Image.open(condition_image_path).convert("RGB")
        condition_images = [("deg", degraded)]
    reference_images = [Image.open(path).convert("RGB") for path in (reference_image_paths or [])]
    target_image = None
    if target_image_path is not None:
        target_image = Image.open(target_image_path).convert("RGB")
    return make_comparison_row(
        condition_images=condition_images,
        reference_images=reference_images,
        target_image=target_image,
        variant_images=[(output_label, output_image)],
        resolution=resolution,
    )


def list_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def select_reference_images(
    *,
    reference_images: list[Path],
    mode: str,
    max_reference_images: int = 3,
) -> list[Path]:
    if mode not in CASE_MODES:
        raise ValueError(f"Unsupported mode: {mode!r}. Expected one of {CASE_MODES}.")
    if mode == "blind":
        return []
    if mode == "ref-single":
        return reference_images[:1]
    return reference_images[:max_reference_images]


def load_manifest(manifest_path: str | Path) -> dict:
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text())
    if not isinstance(data, dict) or "items" not in data or not isinstance(data["items"], list):
        raise ValueError(f"Manifest must be a JSON object containing an 'items' list: {manifest_path}")
    return data
