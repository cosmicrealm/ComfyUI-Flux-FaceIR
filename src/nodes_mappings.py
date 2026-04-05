from __future__ import annotations

from .nodes.apply_lora import FluxFaceIRApplyLoRA
from .nodes.restore_face import FluxFaceIRRestoreFace


NODE_CLASS_MAPPINGS = {
    "FluxFaceIRApplyLoRA": FluxFaceIRApplyLoRA,
    "FluxFaceIRRestoreFace": FluxFaceIRRestoreFace,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxFaceIRApplyLoRA": "Flux FaceIR Apply LoRA",
    "FluxFaceIRRestoreFace": "Flux FaceIR Restore Face",
}
