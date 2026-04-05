from __future__ import annotations

__all__ = [
    "DEFAULT_LORA_PATH",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_OUTPUT_DIR",
    "build_comparison_row",
    "create_pipeline",
    "run_inference",
]


def __getattr__(name: str):
    if name in __all__:
        from .pipeline import (
            DEFAULT_LORA_PATH,
            DEFAULT_MODEL_NAME,
            DEFAULT_OUTPUT_DIR,
            build_comparison_row,
            create_pipeline,
            run_inference,
        )

        exports = {
            "DEFAULT_LORA_PATH": DEFAULT_LORA_PATH,
            "DEFAULT_MODEL_NAME": DEFAULT_MODEL_NAME,
            "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
            "build_comparison_row": build_comparison_row,
            "create_pipeline": create_pipeline,
            "run_inference": run_inference,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
