from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


DEFAULT_PROMPT = "restore a high quality portrait photo of a person, natural skin, detailed face"


def ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor with 3 or 4 dims, got shape {tuple(image.shape)}")
    return image


def comfy_images_to_numpy_rgb(images: torch.Tensor) -> list[np.ndarray]:
    batch = ensure_image_batch(images).detach().cpu().clamp(0.0, 1.0).numpy()
    arrays: list[np.ndarray] = []
    for image in batch:
        image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
        arrays.append(image_u8)
    return arrays


def numpy_rgb_to_comfy_image(image_rgb: np.ndarray) -> torch.Tensor:
    image = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
    return image.unsqueeze(0)


def numpy_rgb_list_to_comfy_image(images: list[np.ndarray]) -> torch.Tensor:
    if not images:
        raise ValueError("Expected at least one image")
    tensors = [torch.from_numpy(image.astype(np.float32) / 255.0) for image in images]
    return torch.stack(tensors, dim=0)


def numpy_mask_list_to_comfy_mask(masks: list[np.ndarray]) -> torch.Tensor:
    if not masks:
        raise ValueError("Expected at least one mask")
    tensors = [torch.from_numpy(mask.astype(np.float32)) for mask in masks]
    return torch.stack(tensors, dim=0)


def comfy_image_to_model_batch(images: torch.Tensor) -> torch.Tensor:
    batch = ensure_image_batch(images).to(dtype=torch.float32)
    return batch.permute(0, 3, 1, 2) * 2.0 - 1.0


def pil_to_comfy_image(image: Image.Image) -> torch.Tensor:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return numpy_rgb_to_comfy_image(rgb)


def normalize_context_batch(face_context: Any, expected: int) -> list[dict[str, Any]]:
    if isinstance(face_context, list):
        contexts = face_context
    else:
        contexts = [face_context]
    if len(contexts) == 1 and expected > 1:
        contexts = contexts * expected
    if len(contexts) != expected:
        raise ValueError(f"FACEIR_CONTEXT length {len(contexts)} does not match batch size {expected}")
    return contexts


def combine_reference_batches(*references) -> torch.Tensor | None:
    batches: list[torch.Tensor] = []
    for reference in references:
        if reference is None:
            continue
        batches.append(ensure_image_batch(reference).to(dtype=torch.float32))
    if not batches:
        return None
    return torch.cat(batches, dim=0)
