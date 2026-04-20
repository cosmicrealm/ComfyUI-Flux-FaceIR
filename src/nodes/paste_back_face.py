from __future__ import annotations

from .common import (
    comfy_images_to_numpy_rgb,
    normalize_context_batch,
    numpy_mask_list_to_comfy_mask,
    numpy_rgb_list_to_comfy_image,
)
from ..flux_faceir.face_align import paste_face_back


class FluxFaceIRPasteBackFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "restored_face": ("IMAGE",),
                "align_params": ("FACEIR_ALIGN_PARAMS",),
                "mask_softness": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("restored_image", "paste_mask")
    FUNCTION = "paste"
    CATEGORY = "Flux FaceIR"

    def paste(self, image, restored_face, align_params, mask_softness):
        source_images = comfy_images_to_numpy_rgb(image)
        restored_faces = comfy_images_to_numpy_rgb(restored_face)
        if len(source_images) != len(restored_faces):
            raise ValueError(
                f"IMAGE batch size {len(source_images)} does not match restored face batch size {len(restored_faces)}."
            )
        params_batch = normalize_context_batch(align_params, len(source_images))

        composed_images = []
        masks = []
        for source_image, restored_face_image, params in zip(source_images, restored_faces, params_batch):
            composed_image, mask = paste_face_back(
                source_image,
                restored_face_image,
                params,
                mask_softness=mask_softness,
            )
            composed_images.append(composed_image)
            masks.append(mask)

        return (
            numpy_rgb_list_to_comfy_image(composed_images),
            numpy_mask_list_to_comfy_mask(masks),
        )
