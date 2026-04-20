from __future__ import annotations

from ..flux_faceir.comfy_paths import get_retinaface_choices, resolve_retinaface_path
from ..flux_faceir.retinaface_runtime import prepare_retinaface_model


class FluxFaceIRLoadRetinaFace:
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = get_retinaface_choices()
        default_name = "retinaface_r34.pth" if "retinaface_r34.pth" in model_choices else model_choices[0]
        return {
            "required": {
                "retinaface_name": (model_choices, {"default": default_name}),
                "network": (
                    [
                        "resnet34",
                        "resnet18",
                        "resnet50",
                        "mobilenetv2",
                        "mobilenetv1",
                        "mobilenetv1_0.50",
                        "mobilenetv1_0.25",
                    ],
                    {"default": "resnet34"},
                ),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "manual_retinaface_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("RETINAFACE_MODEL",)
    RETURN_NAMES = ("retinaface_model",)
    FUNCTION = "load"
    CATEGORY = "Flux FaceIR"

    def load(self, retinaface_name, network, device, manual_retinaface_path):
        resolved_path = resolve_retinaface_path(retinaface_name, manual_retinaface_path)
        if not resolved_path:
            raise ValueError("RetinaFace model path is empty. Select a model or provide a manual path.")
        return (prepare_retinaface_model(weights_path=resolved_path, network=network, device_name=device),)
