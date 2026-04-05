from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    DEFAULT_LORA_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    build_comparison_row,
    create_pipeline,
    ensure_dir,
)


DEFAULT_SINGLE_PROMPT = "restore a high quality portrait photo of a person, natural skin, detailed face"
DEFAULT_REFERENCE_PROMPT = "reference-guided face restoration of a high quality portrait photo of a person, natural skin, detailed face"


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--ae_path", type=str, default=None)
    parser.add_argument("--text_encoder_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=DEFAULT_LORA_PATH)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_targets", type=str, default="flux2_klein")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])


def add_sampling_args(parser: argparse.ArgumentParser, *, prompt: str) -> None:
    parser.add_argument("--prompt", type=str, default=prompt)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)


def add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save_compare", action="store_true")


def create_runtime_from_args(args: argparse.Namespace):
    return create_pipeline(
        model=args.model,
        model_dir=args.model_dir,
        transformer_path=args.transformer_path,
        ae_path=args.ae_path,
        text_encoder_path=args.text_encoder_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_targets,
        device=args.device,
        mixed_precision=args.mixed_precision,
    )


def resolve_output_path(
    *,
    output: str | None,
    output_dir: str | None,
    stem: str,
    suffix: str,
) -> Path:
    if output:
        return Path(output)
    return ensure_dir(output_dir or DEFAULT_OUTPUT_DIR) / f"{stem}{suffix}"


def save_image_and_optional_comparison(
    *,
    image,
    output_path: Path,
    save_compare: bool,
    resolution: int,
    condition_image_path: str | Path | None = None,
    reference_image_paths: list[str | Path] | None = None,
    target_image_path: str | Path | None = None,
    output_label: str = "output",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    if not save_compare:
        return
    compare = build_comparison_row(
        resolution=resolution,
        output_image=image,
        condition_image_path=condition_image_path,
        reference_image_paths=reference_image_paths,
        target_image_path=target_image_path,
        output_label=output_label,
    )
    compare.save(output_path.with_name(output_path.stem + "_compare.png"))
