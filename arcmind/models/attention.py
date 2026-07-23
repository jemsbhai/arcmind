"""
Slow path: Tiny exact attention.

Implements a small number of exact self-attention layers that run at
decision rate (1-10 Hz) over a curated context: compressed SSM snapshots
(episodic memory slots) plus the current decision state.

The memory context is bounded and temporally ordered:
- only the newest ``attn_window_size`` valid memory slots are visible;
- learned relative-age embeddings distinguish otherwise permutation-invariant
  memory reads; and
- causal local attention is used within multi-token query sequences.
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
            f"d_model ({config.d_model}) must be divisible by "
            f"num_attn_heads ({config.num_attn_heads})"
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

        query_positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
        key_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        local_causal_mask = (
            (key_positions <= query_positions)
            & (key_positions > query_positions - self.window_size)
        )

        # Memory is pre-trimmed by SlowAttention. It is fully visible to each
        # query, while the query sequence itself uses causal local attention.
        if memory is not None:
            mem_len = memory.shape[1]
            causal_mask = torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool)
            causal_mask[:, mem_len:] = local_causal_mask
            causal_mask[:, :mem_len] = True
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))
        else:
            causal_mask = local_causal_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))

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
        self.memory_age_embedding = nn.Embedding(
            config.num_memory_slots,
            config.d_model,
        )
        nn.init.normal_(self.memory_age_embedding.weight, std=0.02)
        self.layers = nn.ModuleList(
            [SlowAttentionLayer(config) for _ in range(config.num_attn_layers)]
        )

    def _prepare_memory(self, memory: torch.Tensor | None) -> torch.Tensor | None:
        """Apply the bounded window and encode age relative to the query."""
        if memory is None or memory.shape[1] == 0:
            return None

        memory = memory[:, -self.config.attn_window_size :, :]
        if self.config.ablate_temporal_encoding:
            return memory

        memory_len = memory.shape[1]
        # Memory arrives oldest-to-newest. Age zero denotes the newest prior
        # decision state; larger indices denote progressively older states.
        ages = torch.arange(
            memory_len - 1,
            -1,
            -1,
            device=memory.device,
        )
        return memory + self.memory_age_embedding(ages).unsqueeze(0)

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
        memory = self._prepare_memory(memory)
        for layer in self.layers:
            x = layer(x, memory=memory)
        return x
