"""Spatio-temporal forecaster wrapper.

This module combines:
- A spatial encoder (GCN variant)
- A temporal encoder (TCN variant)

Architectures:
- stack:   GCN is applied at each history step, producing (B, N, T, gcn_dim),
           then a per-node TCN produces temporal features -> horizon head.
- parallel: spatial features from last step + temporal features from raw series
            are fused and projected to horizon.

Expected inputs/outputs:
- x: (B, N, history, 1)
- adj: (N, N)
- y_hat: (B, N, horizon, 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from Fusion_ST_variants.models.gcn_variants import (
    DiffusionGCN,
    GCNLayer,
    GCNWithAttention,
    GCNWithGating,
    GraphAdaptiveLearning,
    GraphSAGE,
    GraphWaveletNetwork,
    LightGraphAttentionNetwork,
)
from Fusion_ST_variants.models.tcn_variants import build_tcn


def build_gcn(name: str, input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    """Factory for selecting a GCN variant by name."""
    key = name.strip().lower()

    if key in {"gcn", "d-gcn", "default"}:
        return GCNLayer(input_dim, hidden_dim)

    if key in {"diffusiongcn", "dgcn", "diffusion"}:
        return DiffusionGCN(input_dim, hidden_dim)

    if key in {"gl", "graphadaptivelearning", "adaptive"}:
        return GraphAdaptiveLearning(input_dim, hidden_dim)

    if key in {"sage", "graphsage"}:
        return GraphSAGE(input_dim, hidden_dim)

    if key in {"gwn", "graphwaveletnetwork", "wavelet"}:
        return GraphWaveletNetwork(input_dim, hidden_dim)

    if key in {"gcn+am", "am", "attention"}:
        return GCNWithAttention(input_features=input_dim, gcn_features=hidden_dim, output_features=hidden_dim, dropout=dropout)

    if key in {"gcn+gating", "gating", "gate"}:
        return GCNWithGating(input_features=input_dim, gcn_features=hidden_dim, output_features=hidden_dim, dropout=dropout)

    if key in {"lightgat", "gat"}:
        return LightGraphAttentionNetwork(input_dim, hidden_dim, dropout=dropout)

    raise ValueError(f"Unknown gcn variant: {name}")


@dataclass
class ForecasterConfig:
    """Configuration for `GcnTcnForecaster`. Kept minimal for experiment logging."""
    num_nodes: int
    history: int
    horizon: int
    gcn_variant: str
    tcn_variant: str
    architecture: str
    fusion_method: str
    parallel_spatial_mode: str = "last"
    gcn_dim: int = 32
    tcn_dim: int = 64
    dropout: float = 0.2


class GcnTcnForecaster(nn.Module):
    """GCN+TCN forecasting model supporting stack/parallel + fusion methods."""
    def __init__(self, cfg: ForecasterConfig):
        super().__init__()
        self.num_nodes = int(cfg.num_nodes)
        self.history = int(cfg.history)
        self.horizon = int(cfg.horizon)
        self.architecture = cfg.architecture
        self.fusion_method = cfg.fusion_method
        self.parallel_spatial_mode = getattr(cfg, "parallel_spatial_mode", "last")

        self.gcn_dim = int(cfg.gcn_dim)
        self.tcn_dim = int(cfg.tcn_dim)

        self.gcn = build_gcn(cfg.gcn_variant, input_dim=1, hidden_dim=self.gcn_dim, dropout=cfg.dropout)

        if self.architecture == "stack":
            self.tcn = build_tcn(cfg.tcn_variant, input_dim=self.gcn_dim, hidden_dim=self.tcn_dim,
                                 dropout=cfg.dropout, spatial_dim=self.gcn_dim)
            self.head = nn.Linear(self.tcn_dim, self.horizon)
        elif self.architecture == "parallel":
            self.tcn = build_tcn(cfg.tcn_variant, input_dim=1, hidden_dim=self.tcn_dim,
                                 dropout=cfg.dropout, spatial_dim=self.gcn_dim)

            combined_dim = self.gcn_dim + self.tcn_dim

            if self.fusion_method == "fgm":
                self.gate = nn.Linear(combined_dim, combined_dim)
            elif self.fusion_method == "pmf":
                self.pmf = nn.Parameter(torch.eye(combined_dim))
            elif self.fusion_method == "fam":
                heads = 2 if combined_dim % 2 == 0 else 1
                self.fam_norm = nn.LayerNorm(combined_dim)
                self.attn = nn.MultiheadAttention(embed_dim=combined_dim, num_heads=heads, batch_first=True)
            elif self.fusion_method == "direct":
                pass
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")

            self.head = nn.Linear(combined_dim, self.horizon)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape (B, N, T, F), got {tuple(x.shape)}")

        b, n, t, f = x.shape
        if n != self.num_nodes:
            raise ValueError(f"num_nodes mismatch: x has {n}, model configured for {self.num_nodes}")

        if f != 1:
            raise ValueError(f"Expected feature dim=1, got {f}")

        if adj.dim() != 2:
            raise ValueError(f"Expected adj shape (N, N), got {tuple(adj.shape)}")

        if self.architecture == "stack":
            gcn_outs = []
            for ti in range(t):
                xt = x[:, :, ti, :]  # (B, N, 1)
                g = self.gcn(xt, adj)  # (B, N, gcn_dim)
                g = torch.relu(g)
                gcn_outs.append(g.unsqueeze(2))

            gcn_outs = torch.cat(gcn_outs, dim=2)  # (B, N, T, gcn_dim)

            tcn_in = gcn_outs.reshape(b * n, t, self.gcn_dim)
            tcn_out = self.tcn(tcn_in)
            last = tcn_out[:, -1, :]

            last = self.dropout(last)
            y = self.head(last)
            y = y.view(b, n, self.horizon, 1)
            return y

        if self.parallel_spatial_mode == "last":
            gcn_feat = self.gcn(x[:, :, -1, :], adj)
            gcn_feat = torch.relu(gcn_feat)
        elif self.parallel_spatial_mode == "all_meanpool":
            gcn_outs = []
            for ti in range(t):
                xt = x[:, :, ti, :]  # (B, N, 1)
                g = self.gcn(xt, adj)  # (B, N, gcn_dim)
                g = torch.relu(g)
                gcn_outs.append(g.unsqueeze(2))
            gcn_feat = torch.cat(gcn_outs, dim=2).mean(dim=2)  # (B, N, gcn_dim)
        else:
            raise ValueError(f"Unknown parallel_spatial_mode: {self.parallel_spatial_mode}")

        tcn_in = x.reshape(b * n, t, 1)
        if getattr(self.tcn, "_requires_spatial", False):
            tcn_out = self.tcn(tcn_in, spatial_context=gcn_feat)
        else:
            tcn_out = self.tcn(tcn_in)
        tcn_feat = tcn_out[:, -1, :].view(b, n, self.tcn_dim)

        combined = torch.cat([gcn_feat, tcn_feat], dim=-1)

        if self.fusion_method == "fgm":
            combined = torch.sigmoid(self.gate(combined)) * combined
        elif self.fusion_method == "pmf":
            combined = combined @ self.pmf
        elif self.fusion_method == "fam":
            combined = self.fam_norm(combined)
            combined, _ = self.attn(combined, combined, combined)
        elif self.fusion_method == "direct":
            pass

        combined = self.dropout(combined)
        y = self.head(combined)
        y = y.view(b, n, self.horizon, 1)
        return y
