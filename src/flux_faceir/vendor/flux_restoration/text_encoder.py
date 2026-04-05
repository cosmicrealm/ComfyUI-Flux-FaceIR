"""
Qwen3 text encoder wrapper for FLUX.2-Klein models.

Klein-4B uses Qwen3-4B  → 3 hidden layers of dim 2560 → context dim 7680
Klein-9B uses Qwen3-8B  → 3 hidden layers of dim 4096 → context dim 12288

Output layers [9, 18, 27] are stacked and concatenated:
    (B, L, D) where D = len(OUTPUT_LAYERS) * hidden_dim
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_LAYERS = (9, 18, 27)
MAX_LENGTH = 512

# HuggingFace repo IDs for each Klein variant
QWEN3_REPOS = {
    "4B": "Qwen/Qwen3-4B",
    "8B": "Qwen/Qwen3-8B",
}


class DummyTextEncoder(nn.Module):
    """Small deterministic text encoder used for local validation."""

    def __init__(self, context_dim: int, max_length: int = 32, device: str | torch.device = "cpu"):
        super().__init__()
        self.context_dim = context_dim
        self.max_length = max_length
        self.register_buffer("basis", torch.linspace(0.0, 1.0, context_dim), persistent=False)
        self.to(device)

    @torch.no_grad()
    def forward(self, prompts: list[str]) -> torch.Tensor:
        batch = len(prompts)
        device = self.basis.device
        positions = torch.arange(self.max_length, device=device, dtype=self.basis.dtype).view(1, self.max_length, 1)
        basis = self.basis.view(1, 1, self.context_dim)
        embeds = torch.zeros((batch, self.max_length, self.context_dim), device=device, dtype=self.basis.dtype)

        for idx, prompt in enumerate(prompts):
            seed = max(1, sum(ord(ch) for ch in prompt) % 997)
            phase = seed / 997.0
            embeds[idx] = torch.sin((positions + 1.0) * (basis + phase))

        return embeds


class Qwen3Embedder(nn.Module):
    """
    Wraps a Qwen3 causal LM and extracts stacked hidden states from
    layers [9, 18, 27] as the text context for FLUX.2-Klein.
    """

    def __init__(self, model_id: str, tokenizer_path: str | None = None,
                 torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype
        )
        # tokenizer may live in a sibling directory (e.g. model_root/tokenizer/)
        tok_path = tokenizer_path if tokenizer_path is not None else model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.max_length = MAX_LENGTH

    @torch.no_grad()
    def forward(self, prompts: list[str]) -> torch.Tensor:
        """
        Args:
            prompts: list of text prompts (length B)

        Returns:
            embeds: (B, max_length, context_dim) — stacked hidden states
        """
        all_input_ids = []
        all_masks = []

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            all_input_ids.append(enc["input_ids"])
            all_masks.append(enc["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.model.device)
        attention_mask = torch.cat(all_masks, dim=0).to(self.model.device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Stack layers and concatenate features: (B, L, D)
        stacked = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)
        return rearrange(stacked, "b c l d -> b l (c d)")


def load_qwen3_embedder(
    variant: str,
    device: str | torch.device = "cuda",
    model_path: str | None = None,
    model_dir: str | None = None,
) -> Qwen3Embedder:
    """Load a Qwen3Embedder.

    Priority for model source:
      1. model_path          – direct path to text_encoder directory
      2. model_dir           – parent model directory; text_encoder/ is a subdir
      3. QWEN3_REPOS[variant] – download from HuggingFace

    When loading from a local directory the tokenizer is expected at a
    sibling 'tokenizer/' folder (standard diffusers pipeline layout).
    """
    if model_path is not None:
        # model_path should be the text_encoder/ subdir
        tok_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "tokenizer")
        if not os.path.isdir(tok_path):
            tok_path = model_path   # fallback: tokenizer inside text_encoder dir
        embedder = Qwen3Embedder(model_id=model_path, tokenizer_path=tok_path)
    elif model_dir is not None:
        te_path = os.path.join(model_dir, "text_encoder")
        tok_path = os.path.join(model_dir, "tokenizer")
        if not os.path.isdir(tok_path):
            tok_path = te_path
        embedder = Qwen3Embedder(model_id=te_path, tokenizer_path=tok_path)
    else:
        repo = QWEN3_REPOS[variant]
        embedder = Qwen3Embedder(model_id=repo)
    return embedder.to(device)


def load_text_encoder(
    variant: str,
    device: str | torch.device = "cuda",
    model_path: str | None = None,
    model_dir: str | None = None,
) -> nn.Module:
    if variant.startswith("dummy:"):
        context_dim = int(variant.split(":", 1)[1])
        return DummyTextEncoder(context_dim=context_dim, device=device)
    if variant == "dummy":
        return DummyTextEncoder(context_dim=256, device=device)
    if variant not in QWEN3_REPOS:
        raise NotImplementedError(
            f"Unsupported text encoder variant '{variant}'. This repository currently supports Qwen3 Klein models only."
        )
    return load_qwen3_embedder(
        variant=variant,
        device=device,
        model_path=model_path,
        model_dir=model_dir,
    )
