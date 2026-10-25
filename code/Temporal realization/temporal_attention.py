"""temporal_attention: extracted algorithm components.

See README.txt and SOURCE_MANIFEST.json for scope and documented changes.
No study data, fitted weights or manuscript statistical workflow is included.
"""
from __future__ import annotations
from typing import Dict
import torch.nn.functional as F
from typing import List
from typing import Optional
from typing import Sequence
import torch.nn as nn
import torch

class WindowNormalizedTCNBlock(nn.Module):
    """Temporal block normalized across the supplied window. GroupNorm includes the time dimension, so this block is not strictly causal over an arbitrary complete sequence. For causal use, supply a separate past-only window per prediction. The current GraphTemporalSplitAE.encode path does not call this block."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout_p: float=0.0) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(max(2, kernel_size))
        self.dilation = int(max(1, dilation))
        self.dropout_p = float(max(0.0, dropout_p))
        self.conv = nn.Conv1d(self.channels, self.channels, kernel_size=self.kernel_size, dilation=self.dilation, padding=0)
        self.norm = nn.GroupNorm(1, self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_left = int((self.kernel_size - 1) * self.dilation)
        y = self.conv(F.pad(x, (pad_left, 0)))
        y = F.silu(self.norm(y))
        if self.dropout_p > 0.0:
            y = F.dropout(y, p=self.dropout_p, training=self.training)
        return F.silu(x + y)

class PersonSpecificTemporalAttention(nn.Module):

    def __init__(self, delay_kmin: int, delay_kmax: int, n_persons: int, n_levels: int=4, person_level_ids: Optional[Sequence[int]]=None, person_delta_scale: float=1.0) -> None:
        super().__init__()
        self.delay_kmin = int(max(0, delay_kmin))
        self.delay_kmax = int(max(self.delay_kmin, delay_kmax))
        self.window_len = int(self.delay_kmax - self.delay_kmin + 1)
        self.n_persons = int(max(1, n_persons))
        self.n_levels = int(max(1, n_levels))
        self.person_delta_scale = float(max(0.0, person_delta_scale))
        self.global_logits = nn.Parameter(torch.zeros((self.window_len,), dtype=torch.float32))
        self.person_delta = nn.Embedding(self.n_persons, self.window_len)
        nn.init.zeros_(self.person_delta.weight)
        level_ids = [int(v) for v in list(person_level_ids or [])]
        if len(level_ids) < self.n_persons:
            level_ids = list(level_ids) + [0] * int(self.n_persons - len(level_ids))
        level_ids_arr = torch.as_tensor([int(max(0, min(self.n_levels - 1, int(v)))) for v in level_ids[:self.n_persons]], dtype=torch.long)
        self.register_buffer('person_level_ids', level_ids_arr, persistent=True)
        self.force_uniform_attention = False

    def set_force_uniform_attention(self, enabled: bool) -> None:
        self.force_uniform_attention = bool(enabled)

    def _coerce_person_id(self, person_id: Optional[torch.Tensor], *, device: torch.device) -> torch.Tensor:
        if person_id is None:
            return torch.zeros((1,), device=device, dtype=torch.long)
        person_t = torch.as_tensor(person_id, device=device, dtype=torch.long).reshape(-1)
        if int(person_t.numel()) <= 0:
            return torch.zeros((1,), device=device, dtype=torch.long)
        return torch.clamp(person_t, min=0, max=int(self.n_persons - 1))

    def _coerce_level_id(self, level_id: Optional[torch.Tensor], *, person_t: torch.Tensor, device: torch.device) -> torch.Tensor:
        if level_id is None:
            if int(self.person_level_ids.numel()) > 0:
                return torch.clamp(self.person_level_ids.to(device=device)[person_t], min=0, max=int(self.n_levels - 1))
            return torch.zeros_like(person_t, device=device, dtype=torch.long)
        level_t = torch.as_tensor(level_id, device=device, dtype=torch.long).reshape(-1)
        if int(level_t.numel()) == 1 and int(person_t.numel()) > 1:
            level_t = level_t.expand_as(person_t)
        if int(level_t.numel()) != int(person_t.numel()):
            raise ValueError('level_id must align with person_id.')
        return torch.clamp(level_t, min=0, max=int(self.n_levels - 1))

    def _select_logits(self, person_id: Optional[torch.Tensor]) -> torch.Tensor:
        person_t = self._coerce_person_id(person_id, device=self.global_logits.device)
        return self.global_logits[None, :] + float(self.person_delta_scale) * self.person_delta(person_t)

    def forward(self, person_id: Optional[torch.Tensor], level_id: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        person_t = self._coerce_person_id(person_id, device=self.global_logits.device)
        level_t = self._coerce_level_id(level_id, person_t=person_t, device=self.global_logits.device)
        logits = self._select_logits(person_id=person_t)
        if bool(getattr(self, 'force_uniform_attention', False)):
            weights = torch.full_like(logits, 1.0 / float(max(1, self.window_len)))
        else:
            weights = F.softmax(logits, dim=-1)
        return {'person_id': person_t, 'level_id': level_t, 'logits': logits, 'weights': weights, 'drive_logits': logits, 'grip_logits': logits, 'drive_weights': weights, 'grip_weights': weights}

    def selected_smoothness_loss(self, person_id: Optional[torch.Tensor], ref: torch.Tensor, level_id: Optional[torch.Tensor]=None) -> torch.Tensor:
        person_t = self._coerce_person_id(person_id, device=ref.device)
        selected_params: List[torch.Tensor] = [self.global_logits.to(device=ref.device), self.person_delta(person_t).to(device=ref.device)]
        if not selected_params:
            return ref.new_zeros(())
        loss_terms: List[torch.Tensor] = []
        for param in selected_params:
            if int(param.shape[-1]) <= 1:
                continue
            diff = param[..., 1:] - param[..., :-1]
            loss_terms.append(torch.mean(torch.square(diff.to(device=ref.device, dtype=ref.dtype))))
        if not loss_terms:
            return ref.new_zeros(())
        return torch.stack(loss_terms).mean()

    def person_delta_l2_loss(self, person_id: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        person_t = self._coerce_person_id(person_id, device=ref.device)
        deltas = self.person_delta(person_t).to(device=ref.device, dtype=ref.dtype)
        return torch.mean(torch.square(deltas)) if int(deltas.numel()) > 0 else ref.new_zeros(())

    def level_delta_l2_loss(self, level_id: Optional[torch.Tensor], person_id: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        return ref.new_zeros(())
