"""
Fast path: SSM core.

Implements a stack of selective state-space layers for processing
continuous sensor streams at high frequency. The default implementation
is a pure-PyTorch SSM that runs on CPU/GPU without custom CUDA kernels.
When mamba-ssm is available, it can be swapped for hardware-optimized kernels.

Design rationale:
- SSM provides O(n) time and O(1) per-token decode (no KV cache).
- Selective (input-dependent) gating follows Mamba's design.
- Produces smoother, more physically plausible control outputs than
  Transformers (Tsuji, IEEE Access 2025).
"""

import torch
import torch.nn as nn

from arcmind.config.defaults import ArcMindConfig


class SSMLayer(nn.Module):
    """Single selective state-space layer (pure PyTorch, no custom CUDA)."""

    def __init__(self, config: ArcMindConfig):
        super().__init__()
        self.config = config
        d_inner = config.d_model * config.ssm_expand_factor

        # Input projection (expand)
        self.in_proj = nn.Linear(config.d_model, d_inner * 2, bias=False)

        # Causal depthwise convolution
        self.conv = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=config.ssm_conv_width,
            padding=config.ssm_conv_width - 1,
            groups=d_inner,
        )

        # SSM parameters: input-dependent discretization
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, config.ssm_state_dim + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(d_inner, -1)
            .clone()
        )
        self.B_proj = nn.Linear(d_inner, config.ssm_state_dim, bias=False)
        self.C_proj = nn.Linear(d_inner, config.ssm_state_dim, bias=False)
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection (contract)
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Output tensor, shape (batch, seq_len, d_model).
        """
        residual = x
        batch, seq_len, _ = x.shape

        # Project and split into two branches
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Causal convolution (conv1d expects B, C, L)
        x_conv = self.conv(x_branch.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = torch.silu(x_conv)

        # SSM scan (sequential for correctness; parallel scan is an optimization)
        A = -torch.exp(self.A_log)  # (d_inner, state_dim)
        dt = torch.softplus(self.dt_proj(x_conv))  # (batch, seq_len, d_inner)
        B = self.B_proj(x_conv)  # (batch, seq_len, state_dim)
        C = self.C_proj(x_conv)  # (batch, seq_len, state_dim)

        d_inner = x_conv.shape[-1]
        state_dim = B.shape[-1]

        # Discretize: dA = exp(A * dt), dB = dt * B
        # Sequential scan (correct reference implementation)
        h = torch.zeros(batch, d_inner, state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (batch, d_inner, 1)
            B_t = B[:, t, :].unsqueeze(1)  # (batch, 1, state_dim)
            C_t = C[:, t, :]  # (batch, state_dim)
            x_t = x_conv[:, t, :].unsqueeze(-1)  # (batch, d_inner, 1)

            dA = torch.exp(A.unsqueeze(0) * dt_t)  # (batch, d_inner, state_dim)
            dB = dt_t * B_t  # (batch, d_inner, state_dim)

            h = dA * h + dB * x_t  # (batch, d_inner, state_dim)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # (batch, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)

        # Apply D (skip connection) and gate
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * torch.silu(z)

        # Project back to d_model and add residual
        return self.norm(self.out_proj(y) + residual)


class SSMCore(nn.Module):
    """Stack of SSM layers forming the fast path."""

    def __init__(self, config: ArcMindConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SSMLayer(config) for _ in range(config.num_ssm_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Output tensor, shape (batch, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x)
        return x
