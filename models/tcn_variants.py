"""TCN (temporal) variants used by Fusion_ST_variants.

These modules operate on per-node sequences.

Expected shapes:
- x: (B, T, C_in)
Returns:
- y: (B, T, C_out)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefaultTCN(nn.Module):
    """Two-layer causal Conv1d TCN.

    Uses explicit left padding to preserve sequence length and causality.
    """
    def __init__(self, input_features: int, output_features: int, kernel_size: int = 2, dilation: int = 1, dropout_rate: float = 0.2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.left_pad = (self.kernel_size - 1) * self.dilation

        self.conv1 = nn.Conv1d(input_features, output_features, kernel_size=self.kernel_size, dilation=self.dilation, padding=0)
        self.conv2 = nn.Conv1d(output_features, output_features, kernel_size=self.kernel_size, dilation=self.dilation, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)

        x = F.pad(x, (self.left_pad, 0))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = F.pad(x, (self.left_pad, 0))
        x = self.conv2(x)
        x = self.relu(x)

        x = x.transpose(1, 2)
        return x


class TCNEnhanced(nn.Module):
    """Multi-layer causal TCN with increasing dilations and LayerNorm."""
    def __init__(
        self,
        input_features: int,
        output_features: int,
        num_layers: int = 2,
        kernel_size: int = 2,
        dilation_base: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.left_pads = []
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        dilation = 1
        c_in = int(input_features)
        for _ in range(int(num_layers)):
            left_pad = (int(kernel_size) - 1) * dilation
            self.left_pads.append(int(left_pad))
            self.layers.append(nn.Conv1d(c_in, int(output_features), kernel_size=int(kernel_size), dilation=dilation, padding=0))
            self.norms.append(nn.LayerNorm(int(output_features)))
            c_in = int(output_features)
            dilation *= int(dilation_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)

        for layer, norm, left_pad in zip(self.layers, self.norms, self.left_pads):
            x = F.pad(x, (int(left_pad), 0))
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = norm(x.transpose(1, 2)).transpose(1, 2)

        return x.transpose(1, 2)


class TCNAttention(nn.Module):
    """TCN followed by MultiheadAttention over the time axis."""
    def __init__(self, tcn_module: nn.Module, embed_dim: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.tcn_module = tcn_module
        self.norm = nn.LayerNorm(int(embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim=int(embed_dim), num_heads=int(num_heads), dropout=float(dropout), batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn_module(x)
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return x


class TCNGated(nn.Module):
    """TCN output gated by a learnable sigmoid gate."""
    def __init__(self, tcn_module: nn.Module, dim: int):
        super().__init__()
        self.tcn_module = tcn_module
        self.gate = nn.Sequential(nn.Linear(int(dim), int(dim)), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tcn_module(x)
        g = self.gate(out)
        return out * g


class TCNSpatiallyConditioned(nn.Module):
    """TCN + STech: GCN spatial features condition the TCN via FiLM modulation.

    After the GCN branch computes per-node spatial features (B, N, spatial_dim),
    those features are projected to per-node scale (gamma) and shift (beta)
    vectors that modulate the TCN output at every timestep:

        out = gamma(spatial) * TCN(x) + beta(spatial)

    This is Feature-wise Linear Modulation (FiLM conditioning), which injects
    spatial context to improve spatiotemporal consistency [44, 76].

    In parallel mode the forecaster detects _requires_spatial=True and passes
    gcn_feat as spatial_context. In stack mode spatial_context is None and
    the module falls back to plain DefaultTCN behaviour.
    """
    _requires_spatial: bool = True

    def __init__(
        self,
        input_features: int,
        output_features: int,
        spatial_dim: int,
        kernel_size: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.tcn_module = DefaultTCN(
            input_features, output_features,
            kernel_size=kernel_size, dropout_rate=dropout_rate,
        )
        self.film_scale = nn.Linear(spatial_dim, output_features)
        self.film_shift = nn.Linear(spatial_dim, output_features)

    def forward(self, x: torch.Tensor, spatial_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x               : (B*N, T, input_features)
        spatial_context : (B, N, spatial_dim) from GCN, or None
        Returns         : (B*N, T, output_features)
        """
        out = self.tcn_module(x)  # (B*N, T, output_features)
        if spatial_context is not None:
            ctx   = spatial_context.reshape(out.size(0), -1)  # (B*N, spatial_dim)
            gamma = self.film_scale(ctx).unsqueeze(1)         # (B*N, 1, output_features)
            beta  = self.film_shift(ctx).unsqueeze(1)         # (B*N, 1, output_features)
            out   = gamma * out + beta
        return out


def build_tcn(name: str, input_dim: int, hidden_dim: int, dropout: float, spatial_dim: int = 64) -> nn.Module:
    """Factory for selecting a TCN variant by name.

    Args:
        name        : variant key (see keys below)
        input_dim   : input feature dimension
        hidden_dim  : output/hidden feature dimension
        dropout     : dropout rate
        spatial_dim : GCN output dim, used only by TCNSpatiallyConditioned (STech)
    """
    key = name.strip().lower()

    if key in {"default", "tcn"}:
        return DefaultTCN(input_dim, hidden_dim, dropout_rate=dropout)

    if key in {"enstr", "enhanced"}:
        return TCNEnhanced(input_dim, hidden_dim, dropout_rate=dropout)

    if key in {"ams", "attention"}:
        base = DefaultTCN(input_dim, hidden_dim, dropout_rate=dropout)
        return TCNAttention(base, embed_dim=hidden_dim, num_heads=2, dropout=dropout)

    if key in {"gms", "gating", "gated"}:
        base = DefaultTCN(input_dim, hidden_dim, dropout_rate=dropout)
        return TCNGated(base, dim=hidden_dim)

    if key in {"stech", "tcn+stech", "spatialcond"}:
        return TCNSpatiallyConditioned(
            input_dim, hidden_dim, spatial_dim=spatial_dim, dropout_rate=dropout
        )

    raise ValueError(f"Unknown tcn variant: {name}")
