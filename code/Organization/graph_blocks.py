"""graph_blocks: extracted algorithm components.

See README.txt and SOURCE_MANIFEST.json for scope and documented changes.
No study data, fitted weights or manuscript statistical workflow is included.
"""
from __future__ import annotations
import torch.nn.functional as F
from typing import Optional
from typing import Tuple
import torch.nn as nn
import torch

def _validate_graph(A, H):
    if A.shape[-2:] != (H.shape[-2], H.shape[-2]):
        raise ValueError('Adjacency must be square and match the node count')
    if not bool(torch.isfinite(A).all()):
        raise ValueError('Adjacency must contain finite values')

def _expand_adj(A_hat: torch.Tensor, batch: int) -> torch.Tensor:
    """Expand (N,N) adjacency to (B,N,N) for batched message passing."""
    A = A_hat
    if A.dim() == 2:
        A = A.unsqueeze(0).expand(batch, -1, -1)
    elif A.dim() == 3:
        if A.shape[0] != batch:
            raise ValueError(f'A_hat batch mismatch: {A.shape[0]} != {batch}')
    else:
        raise ValueError('A_hat must be (N,N) or (B,N,N)')
    return A

def _flatten_nodes(X: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int, int]]]:
    """Flatten (B,T,N,C) to (B*T,N,C) so spatial blocks can be shared."""
    if X.dim() == 4:
        B, T, N, C = X.shape
        return (X.reshape(B * T, N, C), (B, T, N))
    if X.dim() == 3:
        return (X, None)
    raise ValueError('X must be (B,N,C) or (B,T,N,C)')

def _unflatten_nodes(H: torch.Tensor, shape: Optional[Tuple[int, int, int]]) -> torch.Tensor:
    """Restore (B,T,N,*) if temporal shape was provided."""
    if shape is None:
        return H
    B, T, N = shape
    return H.view(B, T, N, -1)

class GCNBlock(nn.Module):
    """GCN block with selectable style to keep behavior consistent across modules."""

    def __init__(self, in_ch: int, hidden: int, dropout: float=0.2, style: str='stacked'):
        super().__init__()
        self.style = style
        self.lin1 = nn.Linear(in_ch, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        H, shape = _flatten_nodes(X)
        A = _expand_adj(A_hat.to(H.device), H.shape[0])
        _validate_graph(A, H)
        if self.style == 'stacked':
            H = self.lin1(H)
            H = torch.bmm(A, H)
            H = F.relu(H)
            H = self.drop(H)
            H = self.lin2(H)
            H = torch.bmm(A, H)
            H = F.relu(H)
        elif self.style == 'residual':
            Hs = self.lin1(H)
            Hn = torch.bmm(A, Hs)
            H = F.relu(Hs + Hn)
            H = self.drop(H)
            H = self.lin2(H)
            H = F.relu(H)
        else:
            raise ValueError(f'Unknown GCN style: {self.style}')
        return _unflatten_nodes(H, shape)

class SAGEBlock(nn.Module):
    """GraphSAGE-Mean style block with optional bias control."""

    def __init__(self, in_ch: int, hidden: int, dropout: float=0.2, bias: bool=True):
        super().__init__()
        self.w_self = nn.Linear(in_ch, hidden, bias=bias)
        self.w_nei = nn.Linear(in_ch, hidden, bias=bias)
        self.lin2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        for m in (self.w_self, self.w_nei, self.lin2):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        H, shape = _flatten_nodes(X)
        A = _expand_adj(A_hat.to(H.device), H.shape[0])
        _validate_graph(A, H)
        Hs = self.w_self(H)
        Hn = torch.bmm(A, self.w_nei(H))
        H = F.relu(Hs + Hn)
        H = self.drop(H)
        H = self.lin2(H)
        H = F.relu(H)
        return _unflatten_nodes(H, shape)

class GATBlock(nn.Module):
    """Simple GAT with adjacency mask; supports batched or temporal inputs."""

    def __init__(self, in_ch: int, hidden: int, heads: int=4, dropout: float=0.1):
        super().__init__()
        if heads <= 0:
            raise ValueError('heads must be positive')
        if hidden % heads != 0:
            raise ValueError('hidden must be divisible by heads')
        self.heads = heads
        self.dk = hidden // heads
        self.q = nn.Linear(in_ch, hidden, bias=False)
        self.k = nn.Linear(in_ch, hidden, bias=False)
        self.v = nn.Linear(in_ch, hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        H, shape = _flatten_nodes(X)
        B, N, _ = H.shape
        A = _expand_adj(A_hat.to(H.device), B)
        _validate_graph(A, H)
        if not bool((A > 0).any(dim=-1).all()):
            raise ValueError('Each attention row needs an allowed neighbor; supply self edges for isolated nodes')
        Q = self.q(H).view(B, N, self.heads, self.dk).permute(0, 2, 1, 3)
        K = self.k(H).view(B, N, self.heads, self.dk).permute(0, 2, 1, 3)
        V = self.v(H).view(B, N, self.heads, self.dk).permute(0, 2, 1, 3)
        attn = torch.matmul(Q, K.transpose(-1, -2)) / self.dk ** 0.5
        mask = (A > 0).unsqueeze(1)
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)
        H = torch.matmul(attn, V)
        H = H.permute(0, 2, 1, 3).contiguous().view(B, N, self.heads * self.dk)
        H = self.proj(H)
        H = F.relu(H)
        return _unflatten_nodes(H, shape)
