from __future__ import annotations

import os

import folder_paths


MANUAL_OPTION = "[manual path]"
NONE_OPTION = "[none]"
RETINAFACE_FOLDER_NAME = "face_detectors"


def _ensure_model_folder(folder_name: str, default_subdir: str) -> None:
    default_path = os.path.join(folder_paths.models_dir, default_subdir)
    supported_extensions = set(getattr(folder_paths, "supported_pt_extensions", set()))
    if folder_name in folder_paths.folder_names_and_paths:
        paths, extensions = folder_paths.folder_names_and_paths[folder_name]
        if default_path not in paths:
            folder_paths.add_model_folder_path(folder_name, default_path, is_default=True)
        if not extensions and supported_extensions:
            folder_paths.folder_names_and_paths[folder_name] = (paths, supported_extensions)
        return
    folder_paths.folder_names_and_paths[folder_name] = ([default_path], supported_extensions)


_ensure_model_folder(RETINAFACE_FOLDER_NAME, RETINAFACE_FOLDER_NAME)


def get_lora_choices() -> list[str]:
    return [NONE_OPTION, MANUAL_OPTION, *folder_paths.get_filename_list("loras")]


def get_retinaface_choices() -> list[str]:
    return [MANUAL_OPTION, *folder_paths.get_filename_list(RETINAFACE_FOLDER_NAME)]


def _resolve_model_path(
    *,
    folder_name: str,
    model_name: str,
    manual_model_path: str | None,
    allow_none: bool = False,
) -> str | None:
    if allow_none and model_name == NONE_OPTION:
        return None
    if model_name and model_name != MANUAL_OPTION:
        return folder_paths.get_full_path_or_raise(folder_name, model_name)
    manual_model_path = (manual_model_path or "").strip()
    return manual_model_path or None


def resolve_lora_path(lora_name: str, manual_lora_path: str | None) -> str | None:
    return _resolve_model_path(
        folder_name="loras",
        model_name=lora_name,
        manual_model_path=manual_lora_path,
        allow_none=True,
    )


def resolve_retinaface_path(retinaface_name: str, manual_retinaface_path: str | None) -> str | None:
    return _resolve_model_path(
        folder_name=RETINAFACE_FOLDER_NAME,
        model_name=retinaface_name,
        manual_model_path=manual_retinaface_path,
        allow_none=False,
    )
