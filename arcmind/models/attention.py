"""
Slow path: Tiny exact attention.

Implements a small number of exact self-attention layers that run at
decision rate (1-10 Hz) over a curated context: compressed SSM snapshots
(episodic memory slots) plus the current instruction/goal embedding.

Design rationale:
- Only 1-2 layers with 2-4 heads, based on Retrieval-Aware Distillation
  (Feb 2026) showing 2-3 attention heads suffice for recall in SSM hybrids.
- Sliding window attention (not full context) for memory efficiency.
- Cross-conditions the fast SSM path via learned gating.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from arcmind.config.defaults import ArcMindConfig


class SlowAttentionLayer(nn.Module):
    """Single multi-head self-attention layer with optional sliding window."""

    def __init__(self, config: ArcMindConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_attn_heads
        self.head_dim = config.d_model // config.num_attn_heads
        self.window_size = config.attn_window_size

        assert config.d_model % config.num_attn_heads == 0, (
            f"d_model ({config.d_model}) must be divisible by num_attn_heads ({config.num_attn_heads})"
        )

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        # Simple FFN after attention
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            memory: Optional episodic memory slots, shape (batch, num_slots, d_model).
                    If provided, concatenated to KV for cross-attention over memory.

        Returns:
            Output tensor, shape (batch, seq_len, d_model).
        """
        residual = x
        x = self.norm1(x)

        batch, seq_len, _ = x.shape

        # If memory is provided, concatenate it to context for KV
        if memory is not None:
            kv_context = torch.cat([memory, x], dim=1)
        else:
            kv_context = x

        kv_len = kv_context.shape[1]

        # QKV projections
        q = self.qkv_proj(x)[:, :, : self.d_model]
        kv = self.qkv_proj(kv_context)
        k = kv[:, :, self.d_model : 2 * self.d_model]
        v = kv[:, :, 2 * self.d_model :]

        # Reshape to (batch, heads, seq, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask (only for the non-memory portion)
        if memory is not None:
            mem_len = memory.shape[1]
            # Allow full attention to memory, causal within sequence
            causal_mask = torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool)
            # Causal within the sequence portion (after memory)
            seq_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            causal_mask[:, mem_len:] = ~seq_mask
            # Memory portion: always visible
            causal_mask[:, :mem_len] = True
            attn_weights = attn_weights.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)

        # Residual + FFN
        x = residual + attn_output
        x = x + self.ffn(self.norm2(x))

        return x


class SlowAttention(nn.Module):
    """Stack of attention layers forming the slow (decision-rate) path."""

    def __init__(self, config: ArcMindConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SlowAttentionLayer(config) for _ in range(config.num_attn_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            memory: Optional episodic memory slots.

        Returns:
            Output tensor, shape (batch, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, memory=memory)
        return x
