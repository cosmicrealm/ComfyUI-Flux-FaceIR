from __future__ import annotations

import folder_paths


MANUAL_OPTION = "[manual path]"
NONE_OPTION = "[none]"


def get_lora_choices() -> list[str]:
    return [NONE_OPTION, MANUAL_OPTION, *folder_paths.get_filename_list("loras")]


def resolve_lora_path(lora_name: str, manual_lora_path: str | None) -> str | None:
    if lora_name == NONE_OPTION:
        return None
    if lora_name and lora_name != MANUAL_OPTION:
        return folder_paths.get_full_path_or_raise("loras", lora_name)
    manual_lora_path = (manual_lora_path or "").strip()
    return manual_lora_path or None
