from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_sft
from safetensors.torch import save_file as save_sft
from torch import nn


LEGACY_ATTN_TARGETS = (
    "img_attn.qkv",
    "img_attn.proj",
    "txt_attn.qkv",
    "txt_attn.proj",
)

LEGACY_ATTN_FFN_TARGETS = LEGACY_ATTN_TARGETS + (
    "img_mlp.0",
    "img_mlp.2",
    "txt_mlp.0",
    "txt_mlp.2",
    "linear1",
    "linear2",
)

# FLUX.2-klein recommended defaults mapped to the local architecture:
# - double-stream image-side attention only
# - single-stream fused attention/MLP projections
FLUX2_KLEIN_TARGETS = (
    "img_attn.qkv",
    "img_attn.proj",
    "linear1",
    "linear2",
)

FLUX2_KLEIN_FULL_TARGETS = FLUX2_KLEIN_TARGETS + (
    "txt_attn.qkv",
    "txt_attn.proj",
    "img_mlp.0",
    "img_mlp.2",
    "txt_mlp.0",
    "txt_mlp.2",
)

# Backward-compatible exports.
ATTN_TARGETS = LEGACY_ATTN_TARGETS
ATTN_FFN_TARGETS = LEGACY_ATTN_FFN_TARGETS

TARGET_PRESETS = {
    "attn": LEGACY_ATTN_TARGETS,
    "attn_ffn": LEGACY_ATTN_FFN_TARGETS,
    "flux2_klein": FLUX2_KLEIN_TARGETS,
    "flux2_klein_full": FLUX2_KLEIN_FULL_TARGETS,
}


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.runtime_scale = 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Keep trainable LoRA weights in fp32 even when the frozen backbone runs in bf16/fp16.
        # This matches standard PEFT practice better and avoids quantizing small adapter updates.
        trainable_dtype = torch.float32
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                base_layer.in_features,
                device=base_layer.weight.device,
                dtype=trainable_dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                rank,
                device=base_layer.weight.device,
                dtype=trainable_dtype,
            )
        )

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        for param in self.base_layer.parameters():
            param.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_layer.bias

    def lora_delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge_(self) -> nn.Linear:
        delta = self.lora_delta().to(self.base_layer.weight.dtype)
        self.base_layer.weight.data.add_(delta)
        return self.base_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        update = F.linear(self.dropout(x).to(dtype=self.lora_A.dtype), self.lora_A)
        update = F.linear(update, self.lora_B) * self.scaling * self.runtime_scale
        return base + update.to(dtype=base.dtype)


def _normalize_module_name(module_name: str) -> str:
    return module_name.replace("._fsdp_wrapped_module", "")


def _normalize_lora_key(key: str) -> str:
    return key.replace("._fsdp_wrapped_module", "")


def _get_child_module(parent: nn.Module, child_name: str) -> nn.Module:
    if child_name.isdigit():
        return parent[int(child_name)]  # type: ignore[index]
    return getattr(parent, child_name)


def _set_child_module(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if child_name.isdigit():
        parent[int(child_name)] = module  # type: ignore[index]
    else:
        setattr(parent, child_name, module)


def _resolve_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = _get_child_module(parent, part)
    return parent, parts[-1]


def get_target_patterns(preset: str | list[str]) -> tuple[str, ...]:
    if isinstance(preset, str):
        if preset not in TARGET_PRESETS:
            raise ValueError(f"Unknown LoRA preset '{preset}'")
        return TARGET_PRESETS[preset]
    return tuple(preset)


def inject_lora(
    model: nn.Module,
    target_patterns: str | list[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    patterns = get_target_patterns(target_patterns)
    replaced: list[str] = []

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(module_name.endswith(pattern) for pattern in patterns):
            continue

        parent, child_name = _resolve_parent(model, module_name)
        wrapped = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        _set_child_module(parent, child_name, wrapped)
        replaced.append(module_name)

    if not replaced:
        raise ValueError(f"No Linear modules matched LoRA targets: {patterns}")

    return replaced


def iter_lora_modules(model: nn.Module):
    for module_name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield module_name, module


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for _, module in iter_lora_modules(model):
        params.extend([module.lora_A, module.lora_B])
    return params


def cast_lora_parameters(model: nn.Module, dtype: torch.dtype) -> None:
    for _, module in iter_lora_modules(model):
        module.lora_A.data = module.lora_A.data.to(dtype=dtype)
        module.lora_B.data = module.lora_B.data.to(dtype=dtype)


@contextmanager
def set_lora_scale(model: nn.Module, scale: float):
    modules = list(iter_lora_modules(model))
    previous_scales = [module.runtime_scale for _, module in modules]
    try:
        for _, module in modules:
            module.runtime_scale = scale
        yield
    finally:
        for (_, module), previous_scale in zip(modules, previous_scales):
            module.runtime_scale = previous_scale


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for module_name, module in iter_lora_modules(model):
        module_name = _normalize_module_name(module_name)
        state[f"{module_name}.lora_A.weight"] = module.lora_A.detach().cpu()
        state[f"{module_name}.lora_B.weight"] = module.lora_B.detach().cpu()
        state[f"{module_name}.alpha"] = torch.tensor(module.alpha, dtype=torch.float32)
    return state


def lora_state_dict_from_full_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    extracted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key = _normalize_lora_key(key)
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            extracted[f"{key}.weight"] = value.detach().cpu()
        elif key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"):
            extracted[key] = value.detach().cpu()
    return extracted


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor], strict: bool = True) -> None:
    state_dict = {_normalize_lora_key(key): value for key, value in state_dict.items()}
    expected = set()
    seen = set()

    for module_name, module in iter_lora_modules(model):
        module_name = _normalize_module_name(module_name)
        key_a = f"{module_name}.lora_A.weight"
        key_b = f"{module_name}.lora_B.weight"
        expected.update({key_a, key_b})

        if key_a not in state_dict or key_b not in state_dict:
            if strict:
                raise KeyError(f"Missing LoRA weights for module '{module_name}'")
            continue

        module.lora_A.data.copy_(state_dict[key_a].to(device=module.lora_A.device, dtype=module.lora_A.dtype))
        module.lora_B.data.copy_(state_dict[key_b].to(device=module.lora_B.device, dtype=module.lora_B.dtype))
        seen.update({key_a, key_b})

    if strict:
        extras = {key for key in state_dict if key.endswith(".weight")} - expected
        if extras:
            raise KeyError(f"Unexpected LoRA keys: {sorted(extras)}")


def save_lora_checkpoint(
    model: nn.Module,
    output_dir: str | Path,
    metadata: dict,
    filename: str = "lora_weights.safetensors",
) -> Path:
    return save_lora_state_dict(get_lora_state_dict(model), output_dir, metadata, filename=filename)


def save_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    output_dir: str | Path,
    metadata: dict,
    filename: str = "lora_weights.safetensors",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_sft(state_dict, output_dir / filename)
    with open(output_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return output_dir


def load_lora_checkpoint(path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"LoRA checkpoint path does not exist: {path}. "
            "Expected a checkpoint directory containing lora_weights.safetensors "
            "or a direct path to a .safetensors file."
        )
    if path.is_dir():
        weights_path = path / "lora_weights.safetensors"
        config_path = path / "adapter_config.json"
    else:
        weights_path = path
        config_path = path.with_name("adapter_config.json")

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing LoRA weights file: {weights_path}. "
            "Expected lora_weights.safetensors inside the checkpoint directory."
        )

    state = load_sft(str(weights_path))
    metadata = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return state, metadata


@contextmanager
def temporary_lora(
    model: nn.Module,
    lora_path: str | Path | None,
    *,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_patterns: str | list[str] = "flux2_klein",
):
    if lora_path is None or str(lora_path).strip() == "":
        yield model
        return

    adapter_state, metadata = load_lora_checkpoint(lora_path)
    preset = metadata.get("lora_target_preset", target_patterns)
    rank = int(metadata.get("lora_rank", rank))
    alpha = float(metadata.get("lora_alpha", alpha))
    dropout = float(metadata.get("lora_dropout", dropout))

    patterns = get_target_patterns(preset)
    replaced: list[tuple[nn.Module, str, nn.Module]] = []
    try:
        for module_name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(module_name.endswith(pattern) for pattern in patterns):
                continue

            parent, child_name = _resolve_parent(model, module_name)
            original = _get_child_module(parent, child_name)
            wrapped = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
            _set_child_module(parent, child_name, wrapped)
            replaced.append((parent, child_name, original))

        if not replaced:
            raise ValueError(f"No Linear modules matched LoRA targets: {patterns}")

        load_lora_state_dict(model, lora_state_dict_from_full_state_dict(adapter_state), strict=True)
        yield model
    finally:
        for parent, child_name, original in reversed(replaced):
            _set_child_module(parent, child_name, original)
