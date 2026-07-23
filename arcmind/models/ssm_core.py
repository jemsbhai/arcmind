"""
Fast path: SSM core.

Implements a stack of selective state-space layers for processing
continuous sensor streams at high frequency. The default implementation
is a pure-PyTorch SSM that runs on CPU/GPU without custom CUDA kernels.

Supports two modes:
- forward(): batch processing of full sequences (training)
- step(): single-timestep recurrent inference with persistent state (deployment)

Design rationale:
- SSM provides O(n) time and O(1) per-token decode (no KV cache).
- State transitions use input-dependent discretization.
- step() mode enables real-time streaming at sensor rate.

This is not an implementation of the published Mamba or Mamba-2 blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from arcmind.config.defaults import ArcMindConfig


class SSMLayer(nn.Module):
    """Single selective state-space layer (pure PyTorch, no custom CUDA)."""

    def __init__(self, config: ArcMindConfig):
        super().__init__()
        self.config = config
        self.d_inner = config.d_model * config.ssm_expand_factor
        self.state_dim = config.ssm_state_dim
        self.conv_width = config.ssm_conv_width

        # Input projection (expand)
        self.in_proj = nn.Linear(config.d_model, self.d_inner * 2, bias=False)

        # Causal depthwise convolution
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.ssm_conv_width,
            padding=config.ssm_conv_width - 1,
            groups=self.d_inner,
        )

        # SSM parameters: input-dependent discretization
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, config.ssm_state_dim + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(self.d_inner, -1)
            .clone()
        )
        self.B_proj = nn.Linear(self.d_inner, config.ssm_state_dim, bias=False)
        self.C_proj = nn.Linear(self.d_inner, config.ssm_state_dim, bias=False)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection (contract)
        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=False)
        self.norm = nn.LayerNorm(config.d_model)

        # Persistent recurrent state (not saved in state_dict)
        self._ssm_state: torch.Tensor | None = None
        self._conv_state: torch.Tensor | None = None

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize recurrent state for streaming inference."""
        state_dtype = dtype or self.A_log.dtype
        self._ssm_state = torch.zeros(
            batch_size,
            self.d_inner,
            self.state_dim,
            device=device,
            dtype=state_dtype,
        )
        self._conv_state = torch.zeros(
            batch_size,
            self.d_inner,
            self.conv_width - 1,
            device=device,
            dtype=state_dtype,
        )

    def reset_state(self) -> None:
        """Clear recurrent state."""
        self._ssm_state = None
        self._conv_state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batch forward pass over a full sequence.

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
        x_conv = F.silu(x_conv)

        # SSM scan
        A = -torch.exp(self.A_log)
        dt = F.softplus(self.dt_proj(x_conv))
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        h = torch.zeros(batch, self.d_inner, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)
            B_t = B[:, t, :].unsqueeze(1)
            C_t = C[:, t, :]
            x_t = x_conv[:, t, :].unsqueeze(-1)

            dA = torch.exp(A.unsqueeze(0) * dt_t)
            dB = dt_t * B_t

            h = dA * h + dB * x_t
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)

        # Apply D (skip connection) and gate
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)

        return self.norm(self.out_proj(y) + residual)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single-timestep recurrent inference. Carries state between calls.

        Must call init_state() before first step().

        Args:
            x: Input tensor, shape (batch, d_model).

        Returns:
            Output tensor, shape (batch, d_model).
        """
        assert self._ssm_state is not None, "Call init_state() before step()"

        residual = x

        # Project and split
        xz = self.in_proj(x)  # (batch, d_inner*2)
        x_branch, z = xz.chunk(2, dim=-1)  # each (batch, d_inner)

        # Compute convolution over [history, current_input]
        # conv_state holds the last conv_width-1 inputs
        conv_weight = self.conv.weight.squeeze(1)  # (d_inner, conv_width)
        conv_input = torch.cat(
            [self._conv_state, x_branch.unsqueeze(-1)], dim=-1
        )  # (batch, d_inner, conv_width)
        x_conv = (conv_input * conv_weight.unsqueeze(0)).sum(dim=-1)  # (batch, d_inner)
        if self.conv.bias is not None:
            x_conv = x_conv + self.conv.bias
        x_conv = F.silu(x_conv)

        # Update conv state: shift left, append current input
        if self.conv_width > 1:
            self._conv_state = conv_input[:, :, 1:]
        else:
            self._conv_state = x_branch.new_empty(
                x_branch.shape[0],
                self.d_inner,
                0,
            )

        # SSM step
        A = -torch.exp(self.A_log)  # (d_inner, state_dim)
        dt = F.softplus(self.dt_proj(x_conv))  # (batch, d_inner)

        B = self.B_proj(x_conv)  # (batch, state_dim)
        C = self.C_proj(x_conv)  # (batch, state_dim)

        dt_e = dt.unsqueeze(-1)       # (batch, d_inner, 1)
        B_e = B.unsqueeze(1)          # (batch, 1, state_dim)
        x_e = x_conv.unsqueeze(-1)    # (batch, d_inner, 1)

        dA = torch.exp(A.unsqueeze(0) * dt_e)   # (batch, d_inner, state_dim)
        dB = dt_e * B_e                          # (batch, d_inner, state_dim)

        self._ssm_state = dA * self._ssm_state + dB * x_e
        y = (self._ssm_state * C.unsqueeze(1)).sum(dim=-1)  # (batch, d_inner)

        # D skip + gate
        y = y + x_conv * self.D
        y = y * F.silu(z)

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
        Batch forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Output tensor, shape (batch, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single-timestep recurrent inference through all layers.

        Args:
            x: Input tensor, shape (batch, d_model).

        Returns:
            Output tensor, shape (batch, d_model).
        """
        for layer in self.layers:
            x = layer.step(x)
        return x

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize recurrent state for all layers."""
        for layer in self.layers:
            layer.init_state(batch_size, device, dtype=dtype)

    def reset_state(self) -> None:
        """Clear recurrent state for all layers."""
        for layer in self.layers:
            layer.reset_state()
