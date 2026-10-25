"""multimodal_model: extracted algorithm components.

See README.txt and SOURCE_MANIFEST.json for scope and documented changes.
No study data, fitted weights or manuscript statistical workflow is included.
"""
from __future__ import annotations
from typing import Optional
import torch.nn as nn
import torch
from tactile_components.organization.graph_blocks import GCNBlock as _GCNBlock, SAGEBlock as _GNNBlock, GATBlock as _GATBlock

class ActionGNNSuite(nn.Module):

    def __init__(self, n_nodes: int, imu_dim: int=6, in_ch: int=1, hidden: int=128, spatial: str='gcn', temporal: str='mean', num_classes: int=5, dropout: float=0.2, use_node_attn: bool=False):
        super().__init__()
        self.n_nodes = n_nodes
        self.imu_dim = imu_dim
        self.spatial = spatial
        self.temporal = temporal
        self.use_node_attn = bool(use_node_attn)
        if spatial == 'gcn':
            self.encoder = _GCNBlock(in_ch, hidden, dropout, style='residual')
        elif spatial == 'gnn':
            self.encoder = _GNNBlock(in_ch, hidden, dropout, bias=False)
        elif spatial == 'attention-gnn':
            self.encoder = _GATBlock(in_ch, hidden, heads=4, dropout=dropout)
        else:
            raise ValueError(f'Unknown spatial: {spatial}')
        temporal_in = hidden + imu_dim
        if temporal == 'lstm':
            self.temporal_head = nn.LSTM(input_size=temporal_in, hidden_size=hidden, batch_first=True)
            self.fc = nn.Linear(hidden, num_classes)
        elif temporal == 'attn':
            self.t_attn_w = nn.Linear(temporal_in, temporal_in)
            self.t_attn_u = nn.Linear(temporal_in, 1)
            self.fc = nn.Linear(temporal_in, num_classes)
        elif self.temporal in ('mean', 'last'):
            self.fc = nn.Linear(temporal_in, num_classes)
        else:
            raise ValueError(f'Unknown temporal type: {temporal}')
        if self.use_node_attn:
            self.node_gate = nn.Linear(hidden, 1)
            nn.init.xavier_uniform_(self.node_gate.weight)
            nn.init.zeros_(self.node_gate.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _reduce(self, steps: torch.Tensor) -> torch.Tensor:
        if self.temporal == 'mean':
            return steps.mean(dim=1)
        if self.temporal == 'last':
            return steps[:, -1]
        if self.temporal == 'lstm':
            _, (h, _) = self.temporal_head(steps)
            return h[-1]
        if self.temporal == 'attn':
            U = torch.tanh(self.t_attn_w(steps))
            scores = self.t_attn_u(U).squeeze(-1)
            w = torch.softmax(scores, dim=1).unsqueeze(-1)
            return (w * steps).sum(dim=1)
        raise RuntimeError('unreachable')

    def forward(self, xt: torch.Tensor, xi: torch.Tensor, A_hat: torch.Tensor, node_pos3d: Optional[torch.Tensor]=None, return_feat: bool=False, return_node_attn: bool=False):
        """
        xt : (B,T,N)  触觉时序
        xi : (B,T,6)  IMU 时序
        A_hat    : (N,N)    归一化后的图邻接
        return_feat: 若为 True，则返回 (logits, feat)，其中
                     feat = 时序聚合后的特征 H，用作“水平/风格特征”
        return_node_attn: 若为 True，则返回每帧节点注意力权重 (B,T,N)
        """
        B, T, N = xt.shape
        assert N == self.n_nodes, f'ActionGNNSuite: n_nodes 不匹配 {N} != {self.n_nodes}'
        assert xi.shape[0] == B and xi.shape[1] == T and (xi.shape[2] == self.imu_dim), f'IMU shape expected (B,T,{self.imu_dim}), got {tuple(xi.shape)}'
        xt = xt.float()
        xi = xi.float()
        pos_all = None
        if node_pos3d is not None:
            if node_pos3d.dim() == 2:
                assert node_pos3d.shape == (N, 3), f'node_pos3d shape expected (N,3), got {tuple(node_pos3d.shape)}'
                pos_all = node_pos3d.view(1, 1, N, 3).expand(B, T, N, 3)
            elif node_pos3d.dim() == 3:
                if node_pos3d.shape == (T, N, 3):
                    pos_all = node_pos3d.unsqueeze(0).expand(B, T, N, 3)
                elif node_pos3d.shape == (B, N, 3):
                    pos_all = node_pos3d.unsqueeze(1).expand(B, T, N, 3)
                else:
                    raise ValueError(f'node_pos3d shape expected (T,N,3) or (B,N,3), got {tuple(node_pos3d.shape)}')
            elif node_pos3d.dim() == 4:
                assert node_pos3d.shape == (B, T, N, 3), f'node_pos3d shape expected (B,T,N,3), got {tuple(node_pos3d.shape)}'
                pos_all = node_pos3d
            else:
                raise ValueError(f'node_pos3d dim must be 2/3/4, got {node_pos3d.dim()}')
            pos_all = pos_all.to(xt.device).float()
        step_list = []
        node_attn_list = [] if return_node_attn else None
        for t in range(T):
            Xt = xt[:, t, :].unsqueeze(-1)
            if pos_all is not None:
                pos = pos_all[:, t]
                Xt = torch.cat([Xt, pos], dim=-1)
            Ht = self.encoder(Xt, A_hat)
            if self.use_node_attn:
                node_scores = self.node_gate(Ht).squeeze(-1)
                node_w = torch.softmax(node_scores, dim=1)
                Ht = (node_w.unsqueeze(-1) * Ht).sum(dim=1)
            else:
                if return_node_attn:
                    node_w = torch.full((B, N), 1.0 / float(N), device=Ht.device)
                Ht = Ht.mean(dim=1)
            if return_node_attn:
                node_attn_list.append(node_w)
            zt = torch.cat([Ht, xi[:, t, :]], dim=-1)
            step_list.append(zt)
        steps = torch.stack(step_list, dim=1)
        H = self._reduce(steps)
        logits = self.fc(H)
        if return_node_attn:
            node_attn = torch.stack(node_attn_list, dim=1) if node_attn_list is not None else None
            if return_feat:
                return (logits, H, node_attn)
            return (logits, node_attn)
        if return_feat:
            return (logits, H)
        return logits
