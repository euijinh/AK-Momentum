"""
Llama-2-style decoder-only transformer with muP, as used for the paper's
language-model experiments.

Architecture (fixed across the width family): SwiGLU feed-forward with expansion
ratio 2.67, rotary position embeddings (base 10,000), pre-RMSNorm, no biases on
linear layers, untied input and output embeddings.  Scale enters only through
``d_model``; ``d_head = 64`` and ``n_layers = 24`` are held fixed, which is what
makes the muP transfer a width-only one.

muP (enabled by ``use_mup``) turns on four rules at once:

    init         hidden linear weights ~ N(0, 1/fan_in); embeddings ~ N(0, 1)
    attention    softmax scale 1/d_head, not 1/sqrt(d_head)
    readout      logits scaled by mup_output_mult * mup_base_width / d_model
    residual     out_proj / value initialized with the usual 1/sqrt(2 L) damping

The matching learning-rate rules live in ``ak_momentum.groups``.

Widths used in the paper: d_model 384 (~67M), 1024 (~370M), 1792 (~1B).
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Architecture and muP settings."""

    d_model: int = 384
    n_layers: int = 24
    n_heads: int = 6
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    hidden_mult: float = 2.67          # SwiGLU expansion; d_ffn = int(d_model * 2.67)
    rope_base: float = 10000.0
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = False
    use_mup: bool = True
    mup_base_width: int = 384          # proxy width the hyperparameters were tuned at
    mup_output_mult: float = 1.0       # alpha in the readout scale

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_ffn(self) -> int:
        return int(self.d_model * self.hidden_mult)


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """RoPE (Su et al., 2021) applied to queries and keys.

    ``RoPE(x, pos) = x * cos(pos theta) + rotate_half(x) * sin(pos theta)`` with
    ``theta_i = base^(-2i/d)``.
    """

    def __init__(self, dim: int, base: float = 10000.0, max_position_embeddings: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._cached_seq_len = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._cos_cached is not None and seq_len <= self._cached_seq_len:
            return
        self._cached_seq_len = max(seq_len, self.max_position_embeddings)
        t = torch.arange(self._cached_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """q, k: (batch, seq, heads, head_dim)."""
        seq_len = q.shape[1]
        self._update_cache(seq_len, q.device, q.dtype)
        cos = self._cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        sin = self._sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)


# ---------------------------------------------------------------------------
# Mixers
# ---------------------------------------------------------------------------

class MHA(nn.Module):
    """Causal multi-head self-attention over a fused QKV projection.

    Uses ``scaled_dot_product_attention``, which picks the best available
    backend.  Under muP the softmax scale is ``1/d_head``.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must divide n_heads"
        self.num_heads = config.n_heads
        self.head_dim = config.d_head
        self.Wqkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, base=config.rope_base,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.mup_attn_scale = bool(config.use_mup)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.Wqkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k = self.rotary_emb(q, k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # (b, heads, seq, head_dim)
        scale = (1.0 / self.head_dim) if self.mup_attn_scale else None
        ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        ctx = ctx.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        return self.out_proj(ctx)


class GLU(nn.Module):
    """SwiGLU feed-forward: ``value(base(x) * silu(gate(x)))``.

    ``base`` and ``gate`` read the same normalized input, so they share one
    Gram matrix under AK-MOMENTUM; ``value`` maps d_ffn -> d_model and is the
    layer whose shape selects the through-N contraction order.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = config.d_ffn
        self.base = nn.Linear(config.d_model, hidden, bias=False)
        self.gate = nn.Linear(config.d_model, hidden, bias=False)
        self.value = nn.Linear(hidden, config.d_model, bias=False)
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(self.base(x) * F.silu(self.gate(x))) * self.output_scale


class TransformerBlock(nn.Module):
    """Pre-norm block; both norms are computed in fp32 for stability."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.sequence_mixer = MHA(config)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.state_mixer = GLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sequence_mixer(self.norm1(x.to(torch.float32)).to(x.dtype))
        x = x + self.state_mixer(self.norm2(x.to(torch.float32)).to(x.dtype))
        return x


# ---------------------------------------------------------------------------
# muP initialization
# ---------------------------------------------------------------------------

def _mup_init_weights(module: nn.Module, n_layers: int) -> None:
    """muP init: ``std = 1/sqrt(fan_in)`` for hidden linears, ``1`` for embeddings.

    Residual-path projections (``out_proj``, ``value``) additionally carry the
    ``1/sqrt(2 L)``-style depth damping, and ``gate`` keeps the 2.2/fan_in gain
    that compensates the SiLU gating.  Applied through ``Module.apply``, so the
    name-matched branches below re-draw those tensors after the generic rule.
    """
    if isinstance(module, nn.Linear):
        fan_in = module.weight.shape[1]
        nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=1.0)

    for name, p in module.named_parameters():
        if "out_proj.weight" in name:
            fan_in = p.shape[1]
            nn.init.normal_(p, mean=0.0, std=1.0 / (math.sqrt(fan_in) * math.sqrt(n_layers)))
        elif "base.weight" in name:
            nn.init.normal_(p, mean=0.0, std=math.sqrt(1.0 / p.shape[1]))
        elif "gate.weight" in name:
            nn.init.normal_(p, mean=0.0, std=math.sqrt(2.2 / p.shape[1]))
        elif "value.weight" in name:
            nn.init.normal_(
                p, mean=0.0,
                std=math.sqrt(1.0 / p.shape[1]) / math.sqrt(2.0 * n_layers),
            )


# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------

class LMBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embeddings(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.ln_f(h.to(torch.float32)).to(h.dtype)


class LanguageModel(nn.Module):
    """Decoder-only LM returning ``{'loss': ..., 'logits': ...}``.

    The readout multiplier ``mup_output_mult * mup_base_width / d_model`` keeps
    the softmax dynamics width-invariant; at the proxy width it equals alpha.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone = LMBackbone(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embeddings.weight
        if config.use_mup:
            self.apply(partial(_mup_init_weights, n_layers=config.n_layers))

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                return_logits: bool = False):
        hidden = self.backbone(input_ids)
        logits = self.lm_head(hidden)
        if self.config.use_mup:
            logits = logits * (
                self.config.mup_output_mult * self.config.mup_base_width / self.config.d_model
            )

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            # Cross-entropy in fp32 for stability under bf16 autocast.
            loss = F.cross_entropy(shift_logits.float(), shift_labels)

        out = {"loss": loss}
        if return_logits:
            out["logits"] = logits
        return out

    def num_parameters(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        return sum(p.numel() for p in ps if p.requires_grad or not trainable_only)
