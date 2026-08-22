from __future__ import annotations

import nodes as comfy_nodes
import torch

from ..flux_faceir.runtime import restore_face
from .common import (
    DEFAULT_PROMPT,
    combine_reference_batches,
    ensure_image_batch,
    pil_to_comfy_image,
)


def _resolve_model_device(faceir_model) -> torch.device | None:
    """Best-effort extraction of the compute device used by a ComfyUI MODEL."""
    model = faceir_model.get("model") if isinstance(faceir_model, dict) else faceir_model
    candidates = []
    inner = getattr(model, "model", None)
    if isinstance(inner, torch.nn.Module):
        candidates.append(inner)
    if isinstance(model, dict):
        inner = model.get("model")
        if isinstance(inner, torch.nn.Module):
            candidates.append(inner)
    if isinstance(model, torch.nn.Module):
        candidates.append(model)
    for module in candidates:
        for param in module.parameters():
            return param.device
    for module in candidates:
        for buffer in module.buffers():
            return buffer.device
    return None


class FluxFaceIRRestoreFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "faceir_model": ("FACEIR_MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "face_image": ("IMAGE",),
                "prompt": ("STRING", {"default": DEFAULT_PROMPT, "multiline": True}),
                "resolution": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 8}),
                "num_steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFF, "step": 1}),
            },
            "optional": {
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_face",)
    FUNCTION = "restore"
    CATEGORY = "Flux FaceIR"

    def restore(
        self,
        faceir_model,
        clip,
        vae,
        face_image,
        prompt,
        resolution,
        num_steps,
        guidance_scale,
        seed,
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
    ):
        face_batch = ensure_image_batch(face_image)
        reference_batch = combine_reference_batches(
            reference_image_1,
            reference_image_2,
            reference_image_3,
            target_size=(int(resolution), int(resolution)),
        )
        device = _resolve_model_device(faceir_model)
        if device is not None:
            face_batch = face_batch.to(device)
            if reference_batch is not None:
                reference_batch = reference_batch.to(device)
        outputs = []
        for index in range(face_batch.shape[0]):
            restored = restore_face(
                faceir_model=faceir_model,
                clip=clip,
                vae=vae,
                face_image=face_batch[index : index + 1],
                prompt=prompt,
                resolution=resolution,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=int(seed) + index,
                reference_image=reference_batch,
            )
            outputs.append(pil_to_comfy_image(restored)[0])
        restored_batch = torch.stack(outputs, dim=0)
        preview = comfy_nodes.PreviewImage().save_images(restored_batch, "flux_faceir_restore_preview")
        return {"ui": preview.get("ui", {}), "result": (restored_batch,)}
