"""spatial_factorization: extracted algorithm components.

See README.txt and SOURCE_MANIFEST.json for scope and documented changes.
No study data, fitted weights or manuscript statistical workflow is included.
"""
from __future__ import annotations
import torch.nn as nn
import torch

class GraphSynergyNMF(nn.Module):

    def __init__(self, n_samples: int, n_groups: int, n_primitives: int):
        super().__init__()
        self.act = nn.Softplus()
        self.W_raw = nn.Parameter(torch.randn(n_samples, n_primitives) * 0.01)
        self.H_raw = nn.Parameter(torch.randn(n_primitives, n_groups) * 0.01)

    def forward(self):
        W = self.act(self.W_raw)
        H = self.act(self.H_raw)
        H = H / (H.sum(dim=1, keepdim=True) + 1e-08)
        F_hat = torch.matmul(W, H)
        return (F_hat, W, H)
