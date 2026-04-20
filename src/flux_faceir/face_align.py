from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np


BASE_FACE_TEMPLATE_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)

_BORDER_MODES = {
    "constant": cv2.BORDER_CONSTANT,
    "reflect101": cv2.BORDER_REFLECT101,
    "reflect": cv2.BORDER_REFLECT,
}


def _as_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, list):
        return [_as_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _as_serializable(item) for key, item in value.items()}
    return value


def align_params_to_json(value: Any) -> str:
    return json.dumps(_as_serializable(value), indent=2, sort_keys=True)


def select_face_detection(detections: np.ndarray, selection: str, image_shape: tuple[int, int, int]) -> np.ndarray:
    if detections.shape[0] == 0:
        raise ValueError("No face detections were found.")
    if detections.shape[0] == 1:
        return detections[0]
    if selection == "highest_score":
        return detections[np.argmax(detections[:, 4])]

    centers = np.stack(
        [
            (detections[:, 0] + detections[:, 2]) * 0.5,
            (detections[:, 1] + detections[:, 3]) * 0.5,
        ],
        axis=1,
    )
    if selection == "center":
        image_center = np.array([image_shape[1] / 2.0, image_shape[0] / 2.0], dtype=np.float32)
        distances = np.linalg.norm(centers - image_center[None, :], axis=1)
        return detections[np.argmin(distances)]

    areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
    return detections[np.argmax(areas)]


def get_face_template(
    *,
    face_size: int,
    crop_scale: float = 1.35,
    crop_shift_y: float = 0.0,
) -> np.ndarray:
    if crop_scale < 1.0:
        raise ValueError("crop_scale must be >= 1.0")
    template = BASE_FACE_TEMPLATE_512 * (float(face_size) / 512.0)
    canvas_center = np.array([face_size / 2.0, face_size / 2.0], dtype=np.float32)
    template = canvas_center + (template - canvas_center) / crop_scale
    template[:, 1] += float(crop_shift_y) * float(face_size)
    return template.astype(np.float32)


def align_face(
    image_rgb: np.ndarray,
    detection: np.ndarray,
    *,
    face_size: int = 512,
    crop_scale: float = 1.35,
    crop_shift_y: float = 0.0,
    border_mode: str = "constant",
) -> tuple[np.ndarray, dict[str, Any]]:
    template = get_face_template(face_size=face_size, crop_scale=crop_scale, crop_shift_y=crop_shift_y)
    landmarks = np.asarray(detection[5:15], dtype=np.float32).reshape(5, 2)
    affine_matrix, _ = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)
    if affine_matrix is None:
        raise ValueError("Failed to estimate the affine transform from RetinaFace landmarks.")
    inverse_affine = cv2.invertAffineTransform(affine_matrix)
    cv_border_mode = _BORDER_MODES.get(border_mode)
    if cv_border_mode is None:
        raise ValueError(f"Unsupported border mode: {border_mode}")
    aligned = cv2.warpAffine(
        image_rgb,
        affine_matrix,
        (face_size, face_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv_border_mode,
        borderValue=(132, 133, 135),
    )
    height, width = image_rgb.shape[:2]
    params = {
        "version": 1,
        "image_width": int(width),
        "image_height": int(height),
        "face_width": int(face_size),
        "face_height": int(face_size),
        "crop_scale": float(crop_scale),
        "crop_shift_y": float(crop_shift_y),
        "border_mode": border_mode,
        "score": float(detection[4]),
        "bbox": [float(value) for value in detection[0:4]],
        "landmarks": landmarks.tolist(),
        "template": template.tolist(),
        "affine_matrix": affine_matrix.astype(np.float32).tolist(),
        "inverse_affine": inverse_affine.astype(np.float32).tolist(),
    }
    return aligned, params


def paste_face_back(
    image_rgb: np.ndarray,
    restored_face_rgb: np.ndarray,
    align_params: dict[str, Any],
    *,
    mask_softness: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    image_height, image_width = image_rgb.shape[:2]
    expected_width = int(align_params["image_width"])
    expected_height = int(align_params["image_height"])
    if image_width != expected_width or image_height != expected_height:
        raise ValueError(
            f"Image shape {(image_width, image_height)} does not match align params {(expected_width, expected_height)}."
        )

    face_width = int(align_params["face_width"])
    face_height = int(align_params["face_height"])
    if restored_face_rgb.shape[1] != face_width or restored_face_rgb.shape[0] != face_height:
        restored_face_rgb = cv2.resize(restored_face_rgb, (face_width, face_height), interpolation=cv2.INTER_LINEAR)

    inverse_affine = np.asarray(align_params["inverse_affine"], dtype=np.float32)
    inv_restored = cv2.warpAffine(restored_face_rgb, inverse_affine, (image_width, image_height), flags=cv2.INTER_LINEAR)

    mask = np.ones((face_height, face_width), dtype=np.float32)
    inv_mask = cv2.warpAffine(mask, inverse_affine, (image_width, image_height), flags=cv2.INTER_LINEAR)
    inv_mask = np.clip(inv_mask, 0.0, 1.0)

    erode_size = max(1, int(round(2 * max(mask_softness, 0.5))))
    inv_mask_erosion = cv2.erode(inv_mask, np.ones((erode_size, erode_size), np.uint8))
    total_face_area = max(1.0, float(np.sum(inv_mask_erosion)))
    edge_width = max(1, int(total_face_area ** 0.5) // 20)
    erosion_radius = max(1, int(round(edge_width * 2 * max(mask_softness, 0.5))))
    inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
    blur_radius = max(1, int(round(edge_width * 2 * max(mask_softness, 0.5))))
    blur_kernel = blur_radius * 2 + 1
    inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_kernel, blur_kernel), 0)
    inv_soft_mask = np.clip(inv_soft_mask, 0.0, 1.0)

    composed = inv_soft_mask[:, :, None] * inv_restored.astype(np.float32)
    composed += (1.0 - inv_soft_mask[:, :, None]) * image_rgb.astype(np.float32)
    composed = np.clip(np.rint(composed), 0, 255).astype(np.uint8)
    return composed, inv_soft_mask.astype(np.float32)
