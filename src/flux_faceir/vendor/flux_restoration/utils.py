"""
Utility functions for FLUX.2 training:
  - Latent packing/unpacking (B,C,H,W) ↔ (B,S,C) with position IDs
  - Text packing with position IDs
  - Flow-matching timestep schedule
  - Weight loading helpers (BFL safetensors format)
"""

from __future__ import annotations

import math
import os
import sys

import torch
from einops import rearrange
from safetensors.torch import load_file as load_sft

from .autoencoder import AutoEncoder, AutoEncoderParams
from .model import DebugFlux2Params, Flux2, Flux2Params, Klein4BParams, Klein9BParams


# ---------------------------------------------------------------------------
# Model configs by name
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "flux.2-klein-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "transformer_file": "flux-2-klein-4b.safetensors",
        "ae_file": "ae.safetensors",
        "ae_subdir": "vae",
        "text_encoder_subdir": "text_encoder",
        "tokenizer_subdir": "tokenizer",
        "params": Klein4BParams(),
        "text_encoder_variant": "4B",
        "guidance_distilled": True,
        "defaults": {"guidance": 1.0, "num_steps": 4},
    },
    "flux.2-klein-base-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "transformer_file": "flux-2-klein-base-4b.safetensors",       # BFL single-file name
        "ae_file": "ae.safetensors",                             # BFL format (fallback)
        "ae_subdir": "vae",                                      # diffusers format subdir
        "text_encoder_subdir": "text_encoder",
        "tokenizer_subdir": "tokenizer",
        "params": Klein4BParams(),
        "text_encoder_variant": "4B",
        "guidance_distilled": False,
        "defaults": {"guidance": 4.0, "num_steps": 50},
    },
    "flux.2-klein-base-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
        "transformer_file": "flux-2-klein-base-9b.safetensors",
        "ae_file": "ae.safetensors",
        "ae_subdir": "vae",
        "text_encoder_subdir": "text_encoder",
        "tokenizer_subdir": "tokenizer",
        "params": Klein9BParams(),
        "text_encoder_variant": "8B",
        "guidance_distilled": False,
        "defaults": {"guidance": 4.0, "num_steps": 50},
    },
    "flux.2-dev": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "transformer_file": "flux2-dev.safetensors",
        "ae_file": "ae.safetensors",
        "ae_subdir": "vae",
        "text_encoder_subdir": "text_encoder",
        "tokenizer_subdir": "tokenizer",
        "params": Flux2Params(),
        "text_encoder_variant": "mistral",
        "guidance_distilled": True,
        "defaults": {"guidance": 4.0, "num_steps": 50},
    },
    "flux.2-debug": {
        "repo_id": None,
        "transformer_file": None,
        "ae_file": None,
        "ae_subdir": None,
        "text_encoder_subdir": None,
        "tokenizer_subdir": None,
        "params": DebugFlux2Params(),
        "text_encoder_variant": "dummy:256",
        "guidance_distilled": False,
        "defaults": {"guidance": 2.5, "num_steps": 8},
    },
}


# ---------------------------------------------------------------------------
# Latent pack / unpack
# ---------------------------------------------------------------------------

def pack_latents(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack latents from (B, C, H, W) to (B, H*W, C) with 4-D position IDs.

    Position ID layout: (t=0, h, w, l=0) — matches the FLUX.2 RoPE embedder.

    Returns:
        packed : (B, S, C)   where S = H * W
        ids    : (B, S, 4)   dtype int64
    """
    B = latents.shape[0]
    packed_list, id_list = [], []

    for i in range(B):
        x = latents[i]      # (C, H, W)
        _, H, W = x.shape
        t = torch.zeros(1, device=x.device, dtype=torch.long)
        h = torch.arange(H, device=x.device, dtype=torch.long)
        w = torch.arange(W, device=x.device, dtype=torch.long)
        l = torch.zeros(1, device=x.device, dtype=torch.long)  # noqa: E741

        ids = torch.cartesian_prod(t, h, w, l)          # (H*W, 4)
        flat = rearrange(x, "c h w -> (h w) c")         # (H*W, C)
        packed_list.append(flat)
        id_list.append(ids)

    return torch.stack(packed_list), torch.stack(id_list)


def unpack_latents(packed: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """
    Unpack (B, S, C) back to (B, C, H, W) using position IDs.

    Args:
        packed : (B, S, C)
        ids    : (B, S, 4)  — (t, h, w, l) coordinates

    Returns:
        (B, C, H, W)
    """
    result = []
    for data, pos in zip(packed, ids):
        _, C = data.shape
        h_ids = pos[:, 1]
        w_ids = pos[:, 2]
        H = h_ids.max().item() + 1
        W = w_ids.max().item() + 1

        flat_ids = h_ids * W + w_ids                     # (S,)
        out = torch.zeros((H * W, C), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, C), data)
        result.append(rearrange(out, "(h w) c -> c h w", h=H, w=W))

    return torch.stack(result)


def pack_text(embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build position IDs for text embeddings (B, L, D).

    Position ID layout: (t=0, h=0, w=0, l) — l indexes token position.

    Returns:
        embeds : unchanged (B, L, D)
        ids    : (B, L, 4)   dtype int64
    """
    B, L, _ = embeds.shape
    device = embeds.device
    id_list = []
    for _ in range(B):
        t = torch.zeros(1, device=device, dtype=torch.long)
        h = torch.zeros(1, device=device, dtype=torch.long)
        w = torch.zeros(1, device=device, dtype=torch.long)
        l = torch.arange(L, device=device, dtype=torch.long)  # noqa: E741
        ids = torch.cartesian_prod(t, h, w, l)               # (L, 4)
        id_list.append(ids)
    return embeds, torch.stack(id_list)


# ---------------------------------------------------------------------------
# Flow-matching timestep schedule
# ---------------------------------------------------------------------------

def generalized_time_snr_shift(t: torch.Tensor, mu: float, sigma: float = 1.0) -> torch.Tensor:
    """Sequence-length-dependent time shift for FLUX.2 flow matching."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int = 50) -> float:
    """Empirical mu parameter for the FLUX.2 SNR time shift."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10  = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    """Return a list of (num_steps+1) timesteps in descending order [1..0]."""
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    return generalized_time_snr_shift(timesteps, mu).tolist()


def sample_sigma(
    batch_size: int,
    method: str = "logit_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Sample sigma (noise level) for flow-matching training.

    method = "logit_normal" : sigma ~ sigmoid(N(logit_mean, logit_std))
    method = "uniform"      : sigma ~ Uniform(0, 1)
    method = "mode"         : sigma ~ mode-biased timestep sampling used by diffusers
    """
    if method == "logit_normal":
        u = torch.randn(batch_size, device=device) * logit_std + logit_mean
        return torch.sigmoid(u)
    if method == "mode":
        u = torch.rand(batch_size, device=device)
        return 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        return torch.rand(batch_size, device=device)


def compute_loss_weights(
    sigmas: torch.Tensor,
    weighting_scheme: str = "none",
) -> torch.Tensor:
    if weighting_scheme == "sigma_sqrt":
        return (sigmas**-2.0).float()
    if weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        return (2 / (math.pi * bot)).float()
    return torch.ones_like(sigmas, dtype=torch.float32)


def build_image_condition_ids(
    latents: torch.Tensor,
    time_offset: int = 10,
    seq_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack reference latents with a non-zero time coordinate so FLUX.2 can
    distinguish them from the scene latents.

    Supported inputs:
      - (B, C, H, W): one reference image per sample
      - (B, R, C, H, W): R reference images per sample
    """
    packed, ids, _ = build_grouped_image_condition_ids(latents, time_offset=time_offset, seq_offset=seq_offset)
    return packed, ids


def build_grouped_image_condition_ids(
    latents: torch.Tensor,
    time_offset: int = 10,
    seq_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Same as `build_image_condition_ids`, but also returns grouped packed tokens.

    Returns:
      packed      : (B, total_tokens, C)
      ids         : (B, total_tokens, 4)
      grouped     : (B, R, tokens_per_ref, C)
    """
    if latents.ndim == 4:
        packed, ids = pack_latents(latents)
        ids = ids.clone()
        ids[..., 0] = time_offset
        if seq_offset:
            ids[..., 3] = seq_offset
        return packed, ids, packed.unsqueeze(1)

    if latents.ndim != 5:
        raise ValueError(f"Expected latents with 4 or 5 dims, got shape {tuple(latents.shape)}")

    packed_refs = []
    ref_ids = []
    for ref_index in range(latents.shape[1]):
        packed, ids = pack_latents(latents[:, ref_index])
        ids = ids.clone()
        ids[..., 0] = time_offset + ref_index
        if seq_offset:
            ids[..., 3] = seq_offset + ref_index
        packed_refs.append(packed)
        ref_ids.append(ids)

    grouped_refs = torch.stack(packed_refs, dim=1)
    return torch.cat(packed_refs, dim=1), torch.cat(ref_ids, dim=1), grouped_refs


def _normalize_optional_path(path: str | None) -> str | None:
    if path is None:
        return None
    path = path.strip()
    return path or None


# ---------------------------------------------------------------------------
# Weight loading (BFL single-file safetensors format)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# diffusers → BFL AutoEncoder key remapping
# ---------------------------------------------------------------------------

def _remap_diffusers_ae_keys(sd: dict, num_resolutions: int = 4) -> dict:
    """
    Remap diffusers AutoencoderKLFlux2 state-dict keys to BFL AutoEncoder
    key names so the weights can be loaded with strict=True.

    Key differences:
      encoder.conv_norm_out         → encoder.norm_out
      decoder.conv_norm_out         → decoder.norm_out
      encoder.down_blocks.N.resnets.M  → encoder.down.N.block.M
      encoder.down_blocks.N.downsamplers.0.conv → encoder.down.N.downsample.conv
      encoder.mid_block.resnets.{0,1}  → encoder.mid.{block_1,block_2}
      encoder.mid_block.attentions.0.{group_norm,to_q,to_k,to_v,to_out.0}
                                       → encoder.mid.attn_1.{norm,q,k,v,proj_out}
      decoder.up_blocks.X.*         → decoder.up.{N-1-X}.*  (reversed order)
      decoder.up_blocks.X.upsamplers.0.conv → decoder.up.{N-1-X}.upsample.conv
      decoder.mid_block.*           → decoder.mid.* (same pattern)
      quant_conv.*                  → encoder.quant_conv.*
      post_quant_conv.*             → decoder.post_quant_conv.*
      .conv_shortcut.               → .nin_shortcut.
    """
    import re
    N = num_resolutions  # typically 4

    _ATTN_MAP = [
        ("group_norm.", "norm."),
        ("to_q.",       "q."),
        ("to_k.",       "k."),
        ("to_v.",       "v."),
        ("to_out.0.",   "proj_out."),
    ]

    def _remap_mid(prefix_in: str, prefix_out: str, k: str) -> str | None:
        """Returns remapped key if it starts with prefix_in, else None."""
        if not k.startswith(prefix_in):
            return None
        rest = k[len(prefix_in):]
        if rest.startswith("resnets.0."):
            return f"{prefix_out}block_1.{rest[len('resnets.0.'):]}"
        if rest.startswith("resnets.1."):
            return f"{prefix_out}block_2.{rest[len('resnets.1.'):]}"
        if rest.startswith("attentions.0."):
            attn_rest = rest[len("attentions.0."):]
            for diff_sfx, bfl_sfx in _ATTN_MAP:
                if attn_rest.startswith(diff_sfx):
                    return f"{prefix_out}attn_1.{bfl_sfx}{attn_rest[len(diff_sfx):]}"
        return None

    remapped = {}
    for k, v in sd.items():
        orig_k = k

        # bn stays as-is
        if k.startswith("bn."):
            remapped[k] = v
            continue

        # top-level quant_conv / post_quant_conv → inside encoder / decoder
        if k.startswith("quant_conv."):
            k = "encoder." + k
        elif k.startswith("post_quant_conv."):
            k = "decoder." + k

        # conv_norm_out → norm_out  (both encoder and decoder)
        k = k.replace("encoder.conv_norm_out.", "encoder.norm_out.")
        k = k.replace("decoder.conv_norm_out.", "decoder.norm_out.")

        # conv_shortcut → nin_shortcut
        k = k.replace(".conv_shortcut.", ".nin_shortcut.")

        # encoder down blocks
        m = re.match(r"encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.*)", k)
        if m:
            k = f"encoder.down.{m.group(1)}.block.{m.group(2)}.{m.group(3)}"

        m = re.match(r"encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.(.*)", k)
        if m:
            k = f"encoder.down.{m.group(1)}.downsample.conv.{m.group(2)}"

        # encoder mid block
        mapped = _remap_mid("encoder.mid_block.", "encoder.mid.", k)
        if mapped is not None:
            k = mapped

        # decoder up blocks  (diffusers up_blocks.X → BFL up.{N-1-X})
        m = re.match(r"decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.*)", k)
        if m:
            bfl_idx = N - 1 - int(m.group(1))
            k = f"decoder.up.{bfl_idx}.block.{m.group(2)}.{m.group(3)}"

        m = re.match(r"decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.(.*)", k)
        if m:
            bfl_idx = N - 1 - int(m.group(1))
            k = f"decoder.up.{bfl_idx}.upsample.conv.{m.group(2)}"

        # decoder mid block
        mapped = _remap_mid("decoder.mid_block.", "decoder.mid.", k)
        if mapped is not None:
            k = mapped

        remapped[k] = v

    # Fix attention weight shapes: diffusers stores them as Linear [out, in],
    # but BFL AttnBlock uses Conv2d [out, in, 1, 1].
    _ATTN_CONV_SUFFIXES = ("attn_1.q.weight", "attn_1.k.weight",
                           "attn_1.v.weight", "attn_1.proj_out.weight")
    result = {}
    for k, v in remapped.items():
        if any(k.endswith(sfx) for sfx in _ATTN_CONV_SUFFIXES) and v.ndim == 2:
            v = v.unsqueeze(-1).unsqueeze(-1)   # [C_out, C_in] → [C_out, C_in, 1, 1]
        result[k] = v
    return result


def _hf_download(repo_id: str, filename: str) -> str:
    """Download a file from the HF hub and return its local path."""
    try:
        import huggingface_hub
        return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        print(f"Failed to download {filename} from {repo_id}: {e}")
        sys.exit(1)


def load_transformer(
    model_name: str,
    weight_path: str | None = None,
    device: str | torch.device = "cuda",
) -> Flux2:
    """
    Load a FLUX.2 transformer from a single safetensors file.

    Args:
        model_name  : key in MODEL_CONFIGS (e.g. 'flux.2-klein-base-4b')
        weight_path : local path to .safetensors; downloads from HF if None
        device      : target device
    """
    cfg = MODEL_CONFIGS[model_name.lower()]
    weight_path = _normalize_optional_path(weight_path)
    if weight_path is None:
        env_key = model_name.upper().replace(".", "_").replace("-", "_") + "_PATH"
        weight_path = _normalize_optional_path(os.environ.get(env_key))
        if weight_path is None and cfg["repo_id"] is not None:
            weight_path = _hf_download(cfg["repo_id"], cfg["transformer_file"])

    if weight_path is None:
        return Flux2(cfg["params"]).to(device)

    with torch.device("meta"):
        model = Flux2(cfg["params"]).to(torch.bfloat16)
    sd = load_sft(weight_path, device=str(device))
    model.load_state_dict(sd, strict=True, assign=True)
    return model.to(device)


def load_ae(
    model_name: str,
    weight_path: str | None = None,
    device: str | torch.device = "cuda",
) -> AutoEncoder:
    """
    Load a FLUX.2 AutoEncoder.

    weight_path can be:
      - A direct .safetensors file in BFL single-file format
      - A directory containing 'diffusion_pytorch_model.safetensors'
        (diffusers format — keys are automatically remapped to BFL names)
      - None → tries env var AE_MODEL_PATH, then model_dir's 'vae/' subdir,
               then downloads from HF
    """
    cfg = MODEL_CONFIGS[model_name.lower()]
    weight_path = _normalize_optional_path(weight_path)
    if weight_path is None:
        weight_path = _normalize_optional_path(os.environ.get("AE_MODEL_PATH"))
    if weight_path is None:
        weight_path = _normalize_optional_path(os.environ.get("MODEL_DIR"))
        ae_subdir = cfg.get("ae_subdir", "vae")
        if weight_path and ae_subdir:
            weight_path = os.path.join(weight_path, ae_subdir)
    if weight_path is None and cfg["repo_id"] is not None:
        weight_path = _hf_download(cfg["repo_id"], cfg["ae_file"])
    if weight_path is None:
        return AutoEncoder(AutoEncoderParams()).to(device)

    # If given a directory, look for diffusers-format safetensors inside
    sft_path = weight_path
    is_diffusers_format = False
    if os.path.isdir(weight_path):
        candidate = os.path.join(weight_path, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(candidate):
            raise FileNotFoundError(
                f"Expected 'diffusion_pytorch_model.safetensors' in {weight_path}"
            )
        sft_path = candidate
        is_diffusers_format = True

    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())
    sd = load_sft(sft_path, device=str(device))

    # Detect diffusers format by presence of 'mid_block' keys even if path was a file
    if not is_diffusers_format and any("mid_block" in k for k in sd):
        is_diffusers_format = True

    if is_diffusers_format:
        sd = _remap_diffusers_ae_keys(sd)

    ae.load_state_dict(sd, strict=True, assign=True)
    return ae.to(device)
