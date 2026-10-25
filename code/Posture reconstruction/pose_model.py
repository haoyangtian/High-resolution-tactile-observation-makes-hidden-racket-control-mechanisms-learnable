"""pose_model: extracted algorithm components.

See README.txt and SOURCE_MANIFEST.json for scope and documented changes.
No study data, fitted weights or manuscript statistical workflow is included.
"""
from __future__ import annotations
from typing import Optional
import torch.nn as nn
import torch
from tactile_components.organization.graph_blocks import GCNBlock as _GCNBlock, SAGEBlock as _SAGEBlock, GATBlock as _GATBlock

class GNNSuite(nn.Module):
    """统一：空间（GCN/SAGE/GAT） × 时序（mean/LSTM/Attention） → 2 回归输出。
    用法：GNNSuite(n_nodes, in_ch=1, hidden=128, spatial='gcn', temporal='mean', heads=4)
    """

    def __init__(self, n_nodes: int, in_ch: int=1, hidden: int=128, spatial: str='gcn', temporal: str='mean', heads: int=4, dropout: float=0.2):
        super().__init__()
        self.n_nodes = n_nodes
        self.temporal = temporal.lower()
        s = spatial.lower()
        if s == 'gcn':
            self.encoder = _GCNBlock(in_ch, hidden, dropout)
        elif s in ('gnn', 'sage', 'sage-mean'):
            self.encoder = _SAGEBlock(in_ch, hidden, dropout)
        elif s in ('gat', 'attention-gnn', 'attention'):
            self.encoder = _GATBlock(in_ch, hidden, heads=heads, dropout=dropout)
        else:
            raise ValueError(f'Unknown spatial type: {spatial}')
        if self.temporal == 'lstm':
            self.temporal_head = nn.LSTM(hidden, hidden, batch_first=True)
        elif self.temporal == 'attn':
            self.t_attn_w = nn.Linear(hidden, hidden)
            self.t_attn_u = nn.Linear(hidden, 1)
        elif self.temporal in ('mean', 'last'):
            pass
        else:
            raise ValueError(f'Unknown temporal type: {temporal}')
        self.fc = nn.Linear(hidden, 2)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _temporal_reduce(self, steps: torch.Tensor) -> torch.Tensor:
        """steps: (B,T,H) -> (B,H)"""
        if self.temporal == 'mean':
            return steps.mean(dim=1)
        if self.temporal == 'last':
            return steps[:, -1]
        if self.temporal == 'lstm':
            out, (h, c) = self.temporal_head(steps)
            return h[-1]
        if self.temporal == 'attn':
            U = torch.tanh(self.t_attn_w(steps))
            scores = self.t_attn_u(U).squeeze(-1)
            w = torch.softmax(scores, dim=1).unsqueeze(-1)
            return (w * steps).sum(dim=1)
        raise RuntimeError('unreachable')

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor, xy: Optional[torch.Tensor]=None) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        B, T, N, C = x.shape
        assert N == self.n_nodes, f'GNNSuite: n_nodes 不匹配 {N} != {self.n_nodes}'
        if xy is not None:
            if xy.dim() == 2:
                assert xy.shape == (N, 3), f'xy shape expected (N,3), got {tuple(xy.shape)}'
                pos = xy.view(1, 1, N, 3).expand(B, T, N, 3)
            elif xy.dim() == 3:
                if xy.shape == (T, N, 3):
                    pos = xy.unsqueeze(0).expand(B, T, N, 3)
                elif xy.shape == (B, N, 3):
                    pos = xy.unsqueeze(1).expand(B, T, N, 3)
                else:
                    raise ValueError(f'xy shape expected (T,N,3) or (B,N,3), got {tuple(xy.shape)}')
            elif xy.dim() == 4:
                assert xy.shape == (B, T, N, 3), f'xy shape expected (B,T,N,3), got {tuple(xy.shape)}'
                pos = xy
            else:
                raise ValueError(f'xy dim must be 2/3/4, got {xy.dim()}')
            pos = pos.to(x.device).float()
            x = torch.cat([x, pos], dim=-1)
            C = C + 3
        step_feats = []
        for t in range(T):
            Xt = x[:, t]
            Ht = self.encoder(Xt, A_hat)
            Ht = Ht.mean(dim=1)
            step_feats.append(Ht)
        steps = torch.stack(step_feats, dim=1)
        H = self._temporal_reduce(steps)
        out = self.fc(H)
        return out
