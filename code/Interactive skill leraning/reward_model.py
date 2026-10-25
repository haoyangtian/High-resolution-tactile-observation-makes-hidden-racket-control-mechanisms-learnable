"""reward_model: extracted algorithm components.

See README.txt and SOURCE_MANIFEST.json for scope and documented changes.
No study data, fitted weights or manuscript statistical workflow is included.
"""
from __future__ import annotations
from typing import Tuple
import torch.nn as nn
import torch

class RewardModel(nn.Module):

    def __init__(self, tac_dim: int, imu_dim: int, skill_dim: int, hidden: int=128, skill_emb: int=16):
        super().__init__()
        self.skill_emb = nn.Embedding(skill_dim, skill_emb)
        self.rnn = nn.GRU(input_size=tac_dim + imu_dim + 1 + skill_emb, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, tac: torch.Tensor, imu: torch.Tensor, skill_id: torch.Tensor, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = tac.shape
        if phase.dim() == 2:
            phase = phase.unsqueeze(0).expand(B, -1, -1)
        skill = self.skill_emb(skill_id).unsqueeze(1).expand(B, T, -1)
        x = torch.cat([tac, imu, phase, skill], dim=-1)
        h, _ = self.rnn(x)
        r = self.out(h).squeeze(-1)
        R = r.sum(dim=1)
        return (r, R)

class ImuPriorModel(nn.Module):

    def __init__(self, tac_dim: int, skill_dim: int, hidden: int=128, imu_dim: int=6, skill_emb: int=16):
        super().__init__()
        self.skill_emb = nn.Embedding(skill_dim, skill_emb)
        self.rnn = nn.GRU(input_size=tac_dim + 1 + skill_emb, hidden_size=hidden, batch_first=True)
        self.out_mu = nn.Linear(hidden, imu_dim)
        self.out_logstd = nn.Linear(hidden, imu_dim)

    def forward(self, tac: torch.Tensor, skill_id: torch.Tensor, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = tac.shape
        if phase.dim() == 2:
            phase = phase.unsqueeze(0).expand(B, -1, -1)
        skill = self.skill_emb(skill_id).unsqueeze(1).expand(B, T, -1)
        x = torch.cat([tac, phase, skill], dim=-1)
        h, _ = self.rnn(x)
        mu = self.out_mu(h)
        logstd = self.out_logstd(h)
        return (mu, logstd)
