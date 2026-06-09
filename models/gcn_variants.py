"""GCN (spatial) variants used by Fusion_ST_variants.

Expected shapes:
- x: (B, N, F)
- adj: (N, N) or (B, N, N)

All modules return per-node features shaped (B, N, C).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Single-step graph convolution: A @ X -> Linear -> (optional activation outside)."""
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.fc = nn.Linear(input_features, output_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(adj, x)
        out = self.fc(out)
        return out


class DiffusionGCN(nn.Module):
    """Simple diffusion-style aggregation over k-hop powers of adjacency."""
    def __init__(self, input_features: int, output_features: int, k_hop: int = 3):
        super().__init__()
        self.fc = nn.Linear(input_features, output_features)
        self.k_hop = int(k_hop)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = x
        adj_power = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        for _ in range(self.k_hop):
            adj_power = torch.matmul(adj_power, adj)
            adj_power = adj_power / (torch.sum(adj_power, dim=-1, keepdim=True) + 1e-6)
            out = out + torch.matmul(adj_power, x)
        out = self.fc(out)
        return torch.relu(out)


class GraphAdaptiveLearning(nn.Module):
    """Adaptive adjacency augmentation based on feature similarity."""
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.fc = nn.Linear(input_features, output_features)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adaptive_adj = torch.matmul(x, x.transpose(-1, -2))
        adaptive_adj = torch.softmax(adaptive_adj, dim=-1)

        if adj.dim() == 2:
            adj_b = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        else:
            adj_b = adj

        combined_adj = adj_b + self.alpha * adaptive_adj
        combined_adj = combined_adj / (torch.sum(combined_adj, dim=-1, keepdim=True) + 1e-6)

        x2 = torch.matmul(combined_adj, x)
        x2 = self.fc(x2)
        return torch.relu(x2)


class GraphSAGE(nn.Module):
    """GraphSAGE-style neighbor aggregation and concatenation."""
    def __init__(self, input_features: int, output_features: int, agg_method: str = "mean"):
        super().__init__()
        self.fc = nn.Linear(input_features * 2, output_features)
        self.agg_method = agg_method

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() == 2:
            adj_b = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        else:
            adj_b = adj

        neighbor_agg = torch.matmul(adj_b, x)
        if self.agg_method == "max":
            neighbor_agg = torch.max(neighbor_agg, dim=-1, keepdim=True)[0]
        else:
            neighbor_agg = neighbor_agg / (torch.sum(adj_b, dim=-1, keepdim=True) + 1e-6)

        out = torch.cat([x, neighbor_agg], dim=-1)
        out = self.fc(out)
        return torch.relu(out)


class GraphWaveletNetwork(nn.Module):
    """Graph wavelet-like transform using a Chebyshev approximation of Laplacian powers."""
    def __init__(self, input_features: int, output_features: int, cheb_order: int = 3):
        super().__init__()
        self.fc = nn.Linear(input_features, output_features)
        self.cheb_order = int(cheb_order)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        num_nodes = adj.size(-1)

        degree = adj.sum(dim=-1)
        d_inv_sqrt = torch.pow(degree + 1e-5, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)

        laplacian = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype) - torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        wavelet_transform = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)
        scaled_lap = laplacian / 2.0  # scale eigenvalues [0,2] -> [0,1] so powers stay bounded
        cheb_term = scaled_lap
        for _ in range(self.cheb_order):
            wavelet_transform = wavelet_transform + cheb_term
            cheb_term = torch.matmul(cheb_term, scaled_lap)

        wavelet_transform = wavelet_transform / (wavelet_transform.sum(dim=-1, keepdim=True) + 1e-6)

        x2 = torch.matmul(wavelet_transform.unsqueeze(0), x)
        x2 = self.fc(x2)
        return torch.relu(x2)


class GCNWithAttention(nn.Module):
    """GCN followed by an attention-style mixing masked by adjacency."""
    def __init__(
        self,
        input_features: int,
        gcn_features: int,
        output_features: int,
        num_heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if gcn_features % num_heads != 0:
            raise ValueError("gcn_features must be divisible by num_heads")

        self.gcn = GCNLayer(input_features, gcn_features)
        self.num_heads = int(num_heads)
        self.head_dim = gcn_features // num_heads

        self.query = nn.Linear(gcn_features, gcn_features)
        self.key = nn.Linear(gcn_features, gcn_features)
        self.value = nn.Linear(gcn_features, gcn_features)
        self.fc = nn.Linear(gcn_features, output_features)

        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(output_features)
        self.skip_connection = nn.Linear(input_features, output_features)
        self.norm = nn.LayerNorm(output_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.size()

        gcn_embeddings = self.gcn(x, adj)
        q = self.query(gcn_embeddings)
        k = self.key(gcn_embeddings)
        v = self.value(gcn_embeddings)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_scores = torch.softmax(attention_scores, dim=-1)

        adj_expanded = adj.unsqueeze(0).expand(b, n, n) if adj.dim() == 2 else adj
        attention_scores = attention_scores * adj_expanded
        attention_scores = self.dropout(attention_scores)

        attended = torch.matmul(attention_scores, v)

        out = self.fc(self.dropout(attended))
        out = self.batch_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = out + self.skip_connection(x)
        return self.norm(F.relu(out))


class GCNWithGating(nn.Module):
    """GCN with feature and update gates + skip connection."""
    def __init__(self, input_features: int, gcn_features: int, output_features: int, dropout: float = 0.2):
        super().__init__()
        self.gcn = GCNLayer(input_features, gcn_features)
        self.feature_gate = nn.Linear(gcn_features, gcn_features)
        self.update_gate = nn.Linear(gcn_features, gcn_features)
        self.skip_connection = nn.Linear(input_features, output_features)
        self.output_fc = nn.Linear(gcn_features, output_features)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        gcn_embeddings = self.gcn(x, adj)
        feature_gate_scores = torch.sigmoid(self.feature_gate(gcn_embeddings))
        update_gate_scores = torch.sigmoid(self.update_gate(gcn_embeddings))

        gated = feature_gate_scores * gcn_embeddings
        updated = update_gate_scores * (gated + gcn_embeddings)

        out = self.output_fc(self.dropout(updated))
        out = out + self.skip_connection(x)
        return self.norm(F.relu(out))


class LightGraphAttentionNetwork(nn.Module):
    """Lightweight attention on projected node features with adjacency masking."""
    def __init__(self, input_features: int, output_features: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_features, output_features)
        self.attn_proj = nn.Parameter(torch.Tensor(output_features, output_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))
        self.skip = nn.Linear(input_features, output_features)
        self.norm = nn.LayerNorm(output_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_proj)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.size()
        x_proj = self.fc(x)

        scores = torch.matmul(x_proj, self.attn_proj)
        scores = torch.matmul(scores, x_proj.transpose(-2, -1)) / (x_proj.shape[-1] ** 0.5)

        if adj.dim() == 2:
            adj_b = adj.unsqueeze(0).expand(b, n, n)
        else:
            adj_b = adj

        scores = scores.masked_fill(adj_b == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, x_proj) + self.bias
        out = out + self.skip(x)
        return self.norm(F.relu(out))
