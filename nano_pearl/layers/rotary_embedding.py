from functools import lru_cache
import math
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def _default_inv_freq(rotary_dim: int, base: float) -> torch.Tensor:
    return 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))


def _apply_llama3_rope_scaling(inv_freq: torch.Tensor, rope_scaling: dict) -> torch.Tensor:
    factor = float(rope_scaling.get("factor", 1.0) or 1.0)
    if factor == 1.0:
        return inv_freq
    low_freq_factor = float(rope_scaling.get("low_freq_factor", 1.0) or 1.0)
    high_freq_factor = float(rope_scaling.get("high_freq_factor", 4.0) or 4.0)
    old_context_len = float(
        rope_scaling.get(
            "original_max_position_embeddings",
            rope_scaling.get("original_max_position", 8192),
        )
        or 8192
    )
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    scaled_inv_freq = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smooth_factor = torch.clamp(smooth_factor, 0.0, 1.0)
    smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / factor) + smooth_factor * inv_freq
    is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, scaled_inv_freq)


class LlamaRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: dict | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = _default_inv_freq(rotary_dim, base)
        if rope_scaling:
            rope_type = str(rope_scaling.get("rope_type", rope_scaling.get("type", "")))
            if rope_type == "llama3":
                inv_freq = _apply_llama3_rope_scaling(inv_freq, rope_scaling)
            elif rope_type == "linear":
                factor = float(rope_scaling.get("factor", 1.0) or 1.0)
                inv_freq = inv_freq / factor
            elif rope_type not in ("", "default"):
                raise NotImplementedError(f"Unsupported Llama rope_scaling={rope_scaling}")
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


def get_rope_llama(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    rotary_emb = LlamaRotaryEmbedding(head_size, rotary_dim, max_position, base, rope_scaling)
    return rotary_emb
