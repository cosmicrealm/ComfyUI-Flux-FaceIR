from __future__ import annotations

from ..flux_faceir.comfy_paths import get_lora_choices, resolve_lora_path
from ..flux_faceir.runtime import prepare_faceir_model


class FluxFaceIRApplyLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        lora_choices = [choice for choice in get_lora_choices() if choice not in ("[manual path]", "[none]")]
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (lora_choices, {"default": lora_choices[0]}),
            },
        }

    RETURN_TYPES = ("FACEIR_MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Flux FaceIR"

    def apply(self, model, lora_name):
        resolved_lora_path = resolve_lora_path(lora_name, None)
        return (prepare_faceir_model(model=model, lora_path=resolved_lora_path),)
