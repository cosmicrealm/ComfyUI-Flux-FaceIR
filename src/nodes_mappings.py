from __future__ import annotations

from .nodes.apply_lora import FluxFaceIRApplyLoRA
from .nodes.detect_align_face import FluxFaceIRDetectAlignFace
from .nodes.load_retinaface import FluxFaceIRLoadRetinaFace
from .nodes.paste_back_face import FluxFaceIRPasteBackFace
from .nodes.restore_face import FluxFaceIRRestoreFace


NODE_CLASS_MAPPINGS = {
    "FluxFaceIRApplyLoRA": FluxFaceIRApplyLoRA,
    "FluxFaceIRLoadRetinaFace": FluxFaceIRLoadRetinaFace,
    "FluxFaceIRDetectAlignFace": FluxFaceIRDetectAlignFace,
    "FluxFaceIRRestoreFace": FluxFaceIRRestoreFace,
    "FluxFaceIRPasteBackFace": FluxFaceIRPasteBackFace,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxFaceIRApplyLoRA": "Flux FaceIR Apply LoRA",
    "FluxFaceIRLoadRetinaFace": "Flux FaceIR Load RetinaFace",
    "FluxFaceIRDetectAlignFace": "Flux FaceIR Detect And Align Face",
    "FluxFaceIRRestoreFace": "Flux FaceIR Restore Face",
    "FluxFaceIRPasteBackFace": "Flux FaceIR Paste Restored Face",
}
