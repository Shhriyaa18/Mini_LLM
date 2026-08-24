"""MiniLLM: a small decoder-only transformer.

Architecture is LLaMA-flavoured: pre-norm blocks, rotary position embeddings,
SwiGLU feed-forward, no biases, tied input/output embeddings.

The KV cache is the part worth reading closely. Caching keys and values means
each decode step only computes attention for the single new token instead of
re-running the whole prefix, which turns generation from O(n^2) total work into
O(n). Getting it right requires two things that are easy to miss:

  * RoPE must be applied at the token's *absolute* position. During cached
    decoding the tensor handed to attention has sequence length 1, so rotating
    by "position 0" would place every generated token at the start of the
    sequence.
  * The causal mask must be indexed by absolute query position against absolute
    key position. Slicing the top-left corner of a triangular mask is only
    correct when there is no cache.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (GPT-NeoX / LLaMA layout)."""

    def __init__(self, d_head: int, max_len: int):
        super().__init__()
        inv_freq = 1.0 / (10_000 ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)
        self.max_len = max_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        pos_offset: int = 0,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rotate x: (B, n_heads, T, d_head).

        Single-sequence decoding passes a scalar `pos_offset` and the positions
        are the contiguous range [pos_offset, pos_offset + T). Batched serving
        passes an explicit `positions` tensor of shape (B, T) instead, because
        requests in a batch sit at different points in their own sequences --
        a scalar offset cannot express that.
        """
        B, _, T, _ = x.shape
        if positions is None:
            end = pos_offset + T
            if end > self.max_len:
                raise ValueError(f"position {end} exceeds RoPE cache length {self.max_len}")
            cos = self.cos_cache[pos_offset:end].view(1, 1, T, -1)
            sin = self.sin_cache[pos_offset:end].view(1, 1, T, -1)
        else:
            if int(positions.max()) >= self.max_len:
                raise ValueError("position exceeds RoPE cache length")
            # (B, T) -> (B, T, d_head) -> (B, 1, T, d_head)
            cos = self.cos_cache[positions].unsqueeze(1)
            sin = self.sin_cache[positions].unsqueeze(1)
        return x * cos.to(x.dtype) + self._rotate_half(x) * sin.to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model
        self.dropout_p = cfg.dropout
        self.use_sdpa = cfg.use_sdpa

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(cfg.d_head, max_len=cfg.context_len)

        mask = torch.tril(torch.ones(cfg.context_len, cfg.context_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def clear_cache(self) -> None:
        self.cache_k = self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        pos_offset: int = 0,
        positions: Optional[torch.Tensor] = None,
        cache=None,
        layer_idx: int = 0,
        slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv_proj(x).split(self.d_model, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Rotate at absolute positions before anything touches the cache: keys
        # go into the cache already rotated, so they are never re-rotated.
        q = self.rope(q, pos_offset, positions)
        k = self.rope(k, pos_offset, positions)

        if cache is not None:
            # Serving path: the engine owns the cache and tells us which slot
            # each sequence lives in.
            if positions is None:
                raise ValueError("external cache requires explicit positions")
            k, v = cache.append(layer_idx, slots, k, v, positions)
            T_k = k.shape[2]
            key_pos = torch.arange(T_k, device=x.device).view(1, 1, 1, T_k)
            # A key index *is* its absolute position, because that is where it
            # was written. So causality is just index <= query position.
            mask = key_pos <= positions.view(B, 1, T, 1)
        else:
            if use_cache:
                if self.cache_k is not None:
                    k = torch.cat([self.cache_k, k], dim=2)
                    v = torch.cat([self.cache_v, v], dim=2)
                self.cache_k, self.cache_v = k, v
            T_k = k.shape[2]
            mask = self.causal_mask[pos_offset : pos_offset + T, :T_k].view(1, 1, T, T_k)

        if self.use_sdpa:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            attn = attn.masked_fill(~mask, float("-inf"))
            attn = self.attn_drop(F.softmax(attn, dim=-1))
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class SwiGLU(nn.Module):
    """Gated feed-forward network (LLaMA / PaLM)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, use_cache=False, pos_offset=0, positions=None,
                cache=None, layer_idx=0, slots=None):
        x = x + self.attn(
            self.norm1(x), use_cache=use_cache, pos_offset=pos_offset,
            positions=positions, cache=cache, layer_idx=layer_idx, slots=slots,
        )
        x = x + self.ffn(self.norm2(x))
        return x


class MiniLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: the output projection reuses the embedding matrix.
        # Saves vocab_size * d_model parameters and usually helps small models.
        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 recipe): keeps activation
        # variance from growing with depth.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def clear_kv_cache(self) -> None:
        for block in self.blocks:
            block.attn.clear_cache()

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        pos_offset: int = 0,
        positions: Optional[torch.Tensor] = None,
        cache=None,
        slots: Optional[torch.Tensor] = None,
    ):
        x = self.drop(self.embed(idx))
        for i, block in enumerate(self.blocks):
            x = block(
                x, use_cache=use_cache, pos_offset=pos_offset,
                positions=positions, cache=cache, layer_idx=i, slots=slots,
            )
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive sampling.

        With use_cache=True the prompt is processed once (prefill) and each
        subsequent step feeds a single token. With use_cache=False the whole
        prefix is recomputed every step -- kept as the baseline that the cached
        path is benchmarked against.
        """
        was_training = self.training
        self.eval()
        self.clear_kv_cache()

        try:
            if use_cache:
                # Prefill.
                logits, _ = self(idx, use_cache=True, pos_offset=0)
                pos = idx.shape[1]

                for _ in range(max_new_tokens):
                    if pos >= self.cfg.context_len:
                        break
                    next_token = self._sample(logits[:, -1, :], temperature, top_k)
                    idx = torch.cat([idx, next_token], dim=1)
                    logits, _ = self(next_token, use_cache=True, pos_offset=pos)
                    pos += 1
            else:
                for _ in range(max_new_tokens):
                    idx_cond = idx[:, -self.cfg.context_len :]
                    logits, _ = self(idx_cond)
                    next_token = self._sample(logits[:, -1, :], temperature, top_k)
                    idx = torch.cat([idx, next_token], dim=1)
        finally:
            self.clear_kv_cache()
            if was_training:
                self.train()

        return idx

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: Optional[int]):
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
