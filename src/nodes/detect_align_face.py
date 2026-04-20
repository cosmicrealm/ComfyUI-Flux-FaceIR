from __future__ import annotations

from ..flux_faceir.face_align import align_face, align_params_to_json, select_face_detection
from .common import comfy_images_to_numpy_rgb, numpy_rgb_list_to_comfy_image


class FluxFaceIRDetectAlignFace:
    @staticmethod
    def _ordered_unique(values):
        unique = []
        for value in values:
            if value not in unique:
                unique.append(value)
        return unique

    def _detect_with_fallbacks(
        self,
        detector,
        source_image,
        *,
        conf_threshold,
        pre_nms_topk,
        nms_threshold,
        post_nms_topk,
        resize_short_edge,
    ):
        resize_candidates = self._ordered_unique(
            [
                int(resize_short_edge),
                max(int(resize_short_edge), 960) if resize_short_edge > 0 else 960,
                max(int(resize_short_edge), 1280) if resize_short_edge > 0 else 1280,
            ]
        )
        conf_candidates = self._ordered_unique(
            [
                round(float(conf_threshold), 2),
                round(min(float(conf_threshold), 0.45), 2),
                round(min(float(conf_threshold), 0.30), 2),
            ]
        )

        attempts = []
        for attempt_resize in resize_candidates:
            for attempt_conf in conf_candidates:
                detections = detector.detect(
                    source_image,
                    conf_threshold=attempt_conf,
                    pre_nms_topk=pre_nms_topk,
                    nms_threshold=nms_threshold,
                    post_nms_topk=post_nms_topk,
                    resize_short_edge=attempt_resize,
                )
                attempts.append((attempt_resize, attempt_conf, int(detections.shape[0])))
                if detections.shape[0] > 0:
                    return detections, attempts
        return None, attempts

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "retinaface_model": ("RETINAFACE_MODEL",),
                "image": ("IMAGE",),
                "face_selection": (["largest", "center", "highest_score"], {"default": "largest"}),
                "face_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 8}),
                "crop_scale": ("FLOAT", {"default": 1.35, "min": 1.0, "max": 2.5, "step": 0.05}),
                "crop_shift_y": ("FLOAT", {"default": 0.0, "min": -0.25, "max": 0.25, "step": 0.01}),
                "conf_threshold": ("FLOAT", {"default": 0.6, "min": 0.05, "max": 0.99, "step": 0.01}),
                "pre_nms_topk": ("INT", {"default": 5000, "min": 1, "max": 20000, "step": 1}),
                "nms_threshold": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 0.95, "step": 0.01}),
                "post_nms_topk": ("INT", {"default": 750, "min": 1, "max": 5000, "step": 1}),
                "resize_short_edge": ("INT", {"default": 640, "min": 0, "max": 4096, "step": 8}),
                "border_mode": (["constant", "reflect101", "reflect"], {"default": "constant"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FACEIR_ALIGN_PARAMS", "STRING")
    RETURN_NAMES = ("aligned_face", "align_params", "align_params_json")
    FUNCTION = "detect_and_align"
    CATEGORY = "Flux FaceIR"

    def detect_and_align(
        self,
        retinaface_model,
        image,
        face_selection,
        face_size,
        crop_scale,
        crop_shift_y,
        conf_threshold,
        pre_nms_topk,
        nms_threshold,
        post_nms_topk,
        resize_short_edge,
        border_mode,
    ):
        detector = retinaface_model["detector"]
        source_images = comfy_images_to_numpy_rgb(image)

        aligned_images = []
        align_params = []
        for index, source_image in enumerate(source_images):
            detections, attempts = self._detect_with_fallbacks(
                detector,
                source_image,
                conf_threshold=conf_threshold,
                pre_nms_topk=pre_nms_topk,
                nms_threshold=nms_threshold,
                post_nms_topk=post_nms_topk,
                resize_short_edge=resize_short_edge,
            )
            if detections is None:
                attempt_summary = ", ".join(
                    f"(resize={attempt_resize}, conf={attempt_conf:.2f}, dets={count})"
                    for attempt_resize, attempt_conf, count in attempts
                )
                height, width = source_image.shape[:2]
                raise ValueError(
                    f"No face detected in batch item {index} with image size {width}x{height}. "
                    f"Tried {attempt_summary}. "
                    "If this is the bundled example workflow, replace the default example image with a real face photo."
                )
            selected = select_face_detection(detections, face_selection, source_image.shape)
            aligned_image, params = align_face(
                source_image,
                selected,
                face_size=face_size,
                crop_scale=crop_scale,
                crop_shift_y=crop_shift_y,
                border_mode=border_mode,
            )
            aligned_images.append(aligned_image)
            align_params.append(params)

        params_output = align_params[0] if len(align_params) == 1 else align_params
        return (
            numpy_rgb_list_to_comfy_image(aligned_images),
            params_output,
            align_params_to_json(params_output),
        )
