from __future__ import annotations

from typing import Any

import torch


def _normalize_optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _get_release_ops():
    from .vendor.flux_restoration.inference import sample_image_comfy

    return sample_image_comfy


def prepare_faceir_model(*, model, lora_path: str | None) -> dict[str, Any]:
    return {
        "model": model,
        "lora_path": _normalize_optional_path(lora_path),
    }


@torch.no_grad()
def restore_face(
    *,
    faceir_model: dict[str, Any],
    clip,
    vae,
    face_image: torch.Tensor,
    prompt: str,
    resolution: int,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    reference_image: torch.Tensor | None = None,
):
    sample_image_comfy = _get_release_ops()
    return sample_image_comfy(
        model=faceir_model["model"],
        clip=clip,
        vae=vae,
        prompt=prompt,
        resolution=resolution,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        condition_image=face_image,
        reference_image=reference_image,
        lora_path=faceir_model.get("lora_path"),
    )
