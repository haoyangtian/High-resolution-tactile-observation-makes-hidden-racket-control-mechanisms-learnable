"""control_model: extracted algorithm components.

See README.txt and SOURCE_MANIFEST.json for scope and documented changes.
No study data, fitted weights or manuscript statistical workflow is included.
"""
from __future__ import annotations
from typing import Dict
import torch.nn.functional as F
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import math
import torch.nn as nn
import numpy as np
import torch
from tactile_components.temporal.temporal_attention import WindowNormalizedTCNBlock, PersonSpecificTemporalAttention
STAGE1_POSE_STATE_DIM = 3

def _stage1_encoder_history_required(local_window: int) -> int:
    return int(max(0, int(local_window) - 1))

def _stage1_encoder_frame_valid_mask_np(seq_len: int, local_window: int) -> np.ndarray:
    n_steps = int(max(0, seq_len))
    mask = np.zeros((n_steps,), dtype=np.bool_)
    if n_steps <= 0:
        return mask
    history = _stage1_encoder_history_required(local_window)
    if history <= 0:
        mask[:] = True
    elif history < n_steps:
        mask[history:] = True
    return mask

def _stage1_encoder_frame_valid_mask_torch(seq_len: int, local_window: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(_stage1_encoder_frame_valid_mask_np(seq_len, local_window)).to(device=device)

class BiomimeticImpedanceDecoder(nn.Module):

    def __init__(self, z_drive_dim: int, z_grip_dim: int, pose_state_dim: int, imu_dim: int, hidden_dim: int, grip_logit_bias_init: float=2.5) -> None:
        super().__init__()
        self.z_drive_dim = int(max(1, z_drive_dim))
        self.z_grip_dim = int(max(1, z_grip_dim))
        self.pose_state_dim = int(max(1, pose_state_dim))
        self.imu_dim = int(max(0, imu_dim))
        self.torque_dim = 3
        self.state_dim = int(self.pose_state_dim * 2 + self.imu_dim)
        self.hidden_dim = int(max(16, hidden_dim))
        self.grip_logit_bias_init = float(grip_logit_bias_init)
        self.state_proj = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.torque_head = nn.Sequential(nn.Linear(self.z_drive_dim + self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.torque_dim))
        self.stiffness_head = nn.Sequential(nn.Linear(self.z_grip_dim + self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.pose_state_dim))
        self.damping_head = nn.Sequential(nn.Linear(self.z_grip_dim + self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.pose_state_dim))
        self._init_grip_logit_bias()

    def _init_grip_logit_bias(self) -> None:
        for head in (self.stiffness_head, self.damping_head):
            final_layer = head[-1] if isinstance(head, nn.Sequential) and len(head) > 0 else None
            if isinstance(final_layer, nn.Linear) and final_layer.bias is not None:
                nn.init.constant_(final_layer.bias, float(self.grip_logit_bias_init))

    def forward(self, z_drive_eff: torch.Tensor, z_grip_eff: torch.Tensor, pose_curr: torch.Tensor, velocity_curr: torch.Tensor, imu_curr: Optional[torch.Tensor], drive_gain: torch.Tensor, downward_drift: torch.Tensor, dt: float) -> Dict[str, torch.Tensor]:
        if z_drive_eff.ndim != 3 or z_grip_eff.ndim != 3:
            raise ValueError('aggregated intents must have shape [B, T, D].')
        if pose_curr.ndim != 3 or velocity_curr.ndim != 3:
            raise ValueError('pose_curr and velocity_curr must have shape [B, T, D].')
        if z_drive_eff.shape[:2] != pose_curr.shape[:2] or z_grip_eff.shape[:2] != pose_curr.shape[:2]:
            raise ValueError('intent/state tensors must align on [B, T].')
        if imu_curr is None:
            imu_use = pose_curr.new_zeros((int(pose_curr.shape[0]), int(pose_curr.shape[1]), self.imu_dim))
        else:
            imu_use = torch.as_tensor(imu_curr, device=pose_curr.device, dtype=pose_curr.dtype)
            if imu_use.ndim == 2:
                imu_use = imu_use.unsqueeze(0)
            if imu_use.shape[:2] != pose_curr.shape[:2]:
                raise ValueError('imu_curr must align with pose_curr on [B, T].')
            if int(imu_use.shape[-1]) != self.imu_dim:
                raise ValueError(f'imu_curr must have last dimension {self.imu_dim}.')
        state_in = torch.cat([pose_curr, velocity_curr, imu_use], dim=-1)
        state_feat = self.state_proj(state_in)
        torque_raw = self.torque_head(torch.cat([z_drive_eff, state_feat], dim=-1))
        grip_in = torch.cat([z_grip_eff, state_feat], dim=-1)
        stiffness_logits = self.stiffness_head(grip_in)
        damping_logits = self.damping_head(grip_in)
        gain = torch.clamp(torch.abs(drive_gain.to(device=pose_curr.device, dtype=pose_curr.dtype)), min=1e-06)
        if int(gain.numel()) != int(self.pose_state_dim):
            gain = gain.reshape(-1)[:self.pose_state_dim]
            if int(gain.numel()) < int(self.pose_state_dim):
                gain = F.pad(gain, (0, int(self.pose_state_dim - int(gain.numel()))), value=1.0)
        gain = gain.view(1, 1, -1)
        delta_v_drive = gain * torque_raw[..., :self.pose_state_dim]
        k_grip_logits = 0.5 * (stiffness_logits + damping_logits)
        grip_logit_clamp_min = float(getattr(self, 'grip_logit_clamp_min', -4.0))
        grip_logit_clamp_max = float(getattr(self, 'grip_logit_clamp_max', 5.0))
        if grip_logit_clamp_max < grip_logit_clamp_min:
            grip_logit_clamp_min, grip_logit_clamp_max = (grip_logit_clamp_max, grip_logit_clamp_min)
        k_grip_logits = torch.clamp(k_grip_logits, min=grip_logit_clamp_min, max=grip_logit_clamp_max)
        k_grip = torch.sigmoid(k_grip_logits)
        k_grip_broadcast = k_grip
        blocked_delta_v = k_grip_broadcast * delta_v_drive
        delta_v_pred = delta_v_drive - blocked_delta_v
        velocity_free = velocity_curr + delta_v_drive
        velocity_next_pred = velocity_curr + delta_v_pred
        pose_free = pose_curr + velocity_free
        pose_next_pred = pose_curr + velocity_next_pred
        zero_seq = torch.zeros_like(delta_v_drive)
        dt_use = float(max(1e-06, dt))
        return {'tau_drive': torque_raw, 'tau_pred_seq': torque_raw, 'torque_seq': torque_raw, 'k_grip_logits': k_grip_logits, 'k_grip': k_grip, 'k_grip_broadcast': k_grip_broadcast, 'impedance_stiffness': k_grip_broadcast, 'impedance_damping': zero_seq, 'pose_curr': pose_curr, 'pose_free': pose_free, 'pose_drift': pose_free, 'pose_anchor': pose_next_pred, 'pose_next_pred': pose_next_pred, 'velocity_base_pred': velocity_curr, 'velocity_free': velocity_free, 'delta_v_drive': delta_v_drive, 'impedance_delta_v': blocked_delta_v, 'drift_term': zero_seq, 'relax_term': zero_seq, 'delta_v_pred': delta_v_pred, 'acc_pred': delta_v_pred / dt_use, 'velocity_next_pred': velocity_next_pred}

class _GraphTemporalSplitAE(nn.Module):

    def __init__(self, n_nodes: int, graph_adj: np.ndarray, hidden_dim: int, z_drive_dim: int, z_grip_dim: int, imu_dim: int=6, local_window: int=5, delay_kmin: int=3, delay_kmax: int=8, n_persons: int=1, n_levels: int=4, person_level_ids: Optional[Sequence[int]]=None, context_downsample: int=5, tcn_kernel_size: int=3, tcn_dilations: Sequence[int]=(1,), pose_state_dim: int=STAGE1_POSE_STATE_DIM, input_dropout_p: float=0.2, tcn_dropout_p: float=0.3, head_dropout_p: float=0.0, drive_gain_init: float=0.1, relax_init: float=0.05, downward_drift_fraction_by_dim: Optional[Sequence[float]]=None, grip_gate_mode: str='mlp', grip_logit_clamp_min: float=-4.0, grip_logit_clamp_max: float=5.0, grip_logit_bias_init: float=2.5) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.hidden_dim = int(hidden_dim)
        self.perception_state_dim = int(hidden_dim)
        self.z_drive_dim = int(z_drive_dim)
        self.z_grip_dim = int(z_grip_dim)
        self.latent_synergy_dim = int(self.z_drive_dim)
        self.latent_null_dim = int(self.z_grip_dim)
        self.imu_dim = int(max(1, imu_dim))
        self.local_window = int(max(1, local_window))
        self.delay_kmin = int(max(0, delay_kmin))
        self.delay_kmax = int(max(self.delay_kmin, delay_kmax))
        self.delay_window_len = int(self.delay_kmax - self.delay_kmin + 1)
        self.n_persons = int(max(1, n_persons))
        self.n_levels = int(max(1, n_levels))
        self.context_downsample = int(max(1, context_downsample))
        self.tcn_kernel_size = int(max(2, tcn_kernel_size))
        self.tcn_dilations = tuple((int(max(1, d)) for d in list(tcn_dilations or (1,))))
        self.pose_state_dim = int(max(1, pose_state_dim))
        self.input_dropout_p = float(min(1.0, max(0.0, input_dropout_p)))
        self.tcn_dropout_p = float(min(1.0, max(0.0, tcn_dropout_p)))
        self.head_dropout_p = float(min(1.0, max(0.0, head_dropout_p)))
        self.grip_gate_mode = str(grip_gate_mode or 'mlp').strip().lower()
        self.grip_logit_clamp_min = float(grip_logit_clamp_min)
        self.grip_logit_clamp_max = float(grip_logit_clamp_max)
        self.grip_logit_bias_init = float(grip_logit_bias_init)
        if self.grip_gate_mode not in {'mlp', 'relative_norm', 'grip_norm'}:
            raise ValueError(f"unsupported grip_gate_mode={grip_gate_mode!r}; expected 'mlp', 'relative_norm', or 'grip_norm'.")
        self.input_dropout = nn.Dropout(p=self.input_dropout_p)
        self.register_buffer('graph_adj', torch.as_tensor(graph_adj, dtype=torch.float32))
        self.B_gain = nn.Parameter(torch.full((self.pose_state_dim,), float(max(1e-06, drive_gain_init)), dtype=torch.float32))
        self.register_buffer('downward_drift', torch.zeros((self.pose_state_dim,), dtype=torch.float32))
        self.graphcam_node_activation = nn.Identity()
        self.input_proj = nn.Linear(self.n_nodes * 2, self.hidden_dim)
        self.graph_activation = nn.SiLU()
        self.perception_frame_mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Dropout(p=self.tcn_dropout_p), nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU())
        self.context_conv1 = nn.Conv1d(self.n_nodes * 2, self.hidden_dim, kernel_size=3, padding=1)
        self.context_conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.context_norm1 = nn.GroupNorm(1, self.hidden_dim)
        self.context_norm2 = nn.GroupNorm(1, self.hidden_dim)
        self.context_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.imu_step_proj = nn.Sequential(nn.Linear(self.imu_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.imu_context_proj = nn.Sequential(nn.Linear(self.imu_dim * 2, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.film_mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim * 2), nn.SiLU(), nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2))
        self.tcn_blocks = nn.ModuleList([WindowNormalizedTCNBlock(channels=self.hidden_dim, kernel_size=self.tcn_kernel_size, dilation=int(d), dropout_p=self.tcn_dropout_p) for d in self.tcn_dilations])
        self.intent_projection = nn.Sequential(nn.Linear(self.perception_state_dim, self.hidden_dim), nn.SiLU(), nn.Dropout(p=self.head_dropout_p), nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU())
        self.perception_recon_head = nn.Sequential(nn.Linear(self.perception_state_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.n_nodes), nn.Softplus())
        self.drive_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Dropout(p=self.head_dropout_p), nn.Linear(self.hidden_dim, self.z_drive_dim))
        self.grip_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Dropout(p=self.head_dropout_p), nn.Linear(self.hidden_dim, self.z_grip_dim))
        self.synergy_basis_raw = nn.Parameter(torch.zeros((self.z_drive_dim, self.n_nodes), dtype=torch.float32))
        self.grip_basis_raw = nn.Parameter(torch.zeros((self.z_grip_dim, self.n_nodes), dtype=torch.float32))
        nn.init.xavier_uniform_(self.synergy_basis_raw)
        nn.init.xavier_uniform_(self.grip_basis_raw)
        self.cerebellum_hidden_dim = 64
        self.cerebellum_state_dim = int(self.pose_state_dim * 2)
        self.grip_gate_dim = int(self.pose_state_dim)
        self.cerebellum_grip_input_dim = int(self.z_grip_dim + self.pose_state_dim)
        self.cerebellum_drive_input_dim = int(self.z_drive_dim + self.cerebellum_state_dim)
        self.cerebellum_grip_body = nn.Sequential(nn.Linear(self.cerebellum_grip_input_dim, self.cerebellum_hidden_dim), nn.ReLU(), nn.Dropout(p=self.head_dropout_p))
        self.cerebellum_grip_out = nn.Linear(self.cerebellum_hidden_dim, self.grip_gate_dim)
        self.cerebellum_grip_activation = nn.Sigmoid()
        self.grip_norm_logit_scale_raw = nn.Parameter(torch.as_tensor(float(math.log(math.expm1(1.0))), dtype=torch.float32))
        self.grip_norm_logit_bias = nn.Parameter(torch.zeros((self.grip_gate_dim,), dtype=torch.float32))
        self.cerebellum_drive = nn.Sequential(nn.Linear(self.cerebellum_drive_input_dim, self.cerebellum_hidden_dim), nn.ReLU(), nn.Dropout(p=self.head_dropout_p), nn.Linear(self.cerebellum_hidden_dim, self.pose_state_dim))
        self.person_temporal_attention = PersonSpecificTemporalAttention(delay_kmin=self.delay_kmin, delay_kmax=self.delay_kmax, n_persons=self.n_persons, n_levels=self.n_levels, person_level_ids=person_level_ids)
        self.biomimetic_impedance_decoder = BiomimeticImpedanceDecoder(z_drive_dim=self.z_drive_dim, z_grip_dim=self.z_grip_dim, pose_state_dim=self.pose_state_dim, imu_dim=self.imu_dim, hidden_dim=self.hidden_dim, grip_logit_bias_init=self.grip_logit_bias_init)
        self.biomimetic_impedance_decoder.grip_logit_clamp_min = float(self.grip_logit_clamp_min)
        self.biomimetic_impedance_decoder.grip_logit_clamp_max = float(self.grip_logit_clamp_max)

    def _relative_norm_grip_logits(self, z_drive_now: torch.Tensor, z_grip_now: torch.Tensor) -> torch.Tensor:
        if z_drive_now.ndim != 3 or z_grip_now.ndim != 3 or z_drive_now.shape[:2] != z_grip_now.shape[:2]:
            raise ValueError('z_drive_now and z_grip_now must both have shape [B, S, D] aligned on [B, S].')
        eps = 1e-06
        grip_norm = torch.linalg.vector_norm(z_grip_now, ord=2, dim=-1, keepdim=True).clamp_min(eps)
        drive_norm = torch.linalg.vector_norm(z_drive_now, ord=2, dim=-1, keepdim=True).clamp_min(eps)
        return torch.log(grip_norm) - torch.log(drive_norm)

    def _grip_norm_grip_logits(self, z_grip_now: torch.Tensor) -> torch.Tensor:
        if z_grip_now.ndim != 3:
            raise ValueError('z_grip_now must have shape [B, S, D].')
        eps = 1e-06
        grip_norm = torch.linalg.vector_norm(z_grip_now, ord=2, dim=-1, keepdim=True).clamp_min(eps)
        scale = F.softplus(self.grip_norm_logit_scale_raw).view(1, 1, 1).to(device=z_grip_now.device, dtype=z_grip_now.dtype)
        bias = self.grip_norm_logit_bias.view(1, 1, -1).to(device=z_grip_now.device, dtype=z_grip_now.dtype)
        return scale * torch.log(grip_norm) + bias

    def _spatial_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.input_dropout_p > 0.0:
            x = self.input_dropout(x)
        x_graph = torch.einsum('btn,nm->btm', x, self.graph_adj)
        node_feat = torch.stack([x, x_graph], dim=-1)
        x_graph_2hop = torch.einsum('btn,nm->btm', x_graph, self.graph_adj)
        graphcam_feat = F.silu(torch.stack([x_graph, x_graph_2hop], dim=-1))
        graphcam_feat = self.graphcam_node_activation(graphcam_feat)
        if not self.training:
            node_feat = node_feat + (graphcam_feat - graphcam_feat.detach())
        enc_in = node_feat.reshape(x.shape[0], x.shape[1], self.n_nodes * 2)
        feat = self.graph_activation(self.input_proj(enc_in))
        return (feat, enc_in)

    def _segment_context(self, enc_in: torch.Tensor) -> torch.Tensor:
        coarse = enc_in[:, ::self.context_downsample, :]
        if coarse.shape[1] <= 0:
            coarse = enc_in[:, -1:, :]
        ctx = coarse.transpose(1, 2)
        ctx = F.silu(self.context_norm1(self.context_conv1(ctx)))
        ctx = F.silu(self.context_norm2(self.context_conv2(ctx)))
        ctx = ctx.mean(dim=-1)
        return F.silu(self.context_proj(ctx))

    def _coerce_imu_seq(self, imu: Optional[torch.Tensor], batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.imu_dim <= 0:
            return torch.zeros((batch_size, seq_len, 0), device=device, dtype=dtype)
        if imu is None:
            return torch.zeros((batch_size, seq_len, self.imu_dim), device=device, dtype=dtype)
        imu_t = torch.as_tensor(imu, device=device, dtype=dtype)
        if imu_t.ndim == 2:
            imu_t = imu_t.unsqueeze(0)
        if imu_t.ndim != 3:
            raise ValueError(f'imu must have shape [B, T, {self.imu_dim}].')
        if int(imu_t.shape[-1]) != self.imu_dim:
            raise ValueError(f'imu must have shape [B, T, {self.imu_dim}].')
        if int(imu_t.shape[0]) != batch_size:
            raise ValueError('imu batch size must match tactile batch size.')
        if int(imu_t.shape[1]) < seq_len:
            pad = imu_t[:, -1:, :].expand(batch_size, seq_len - int(imu_t.shape[1]), self.imu_dim)
            imu_t = torch.cat([imu_t, pad], dim=1)
        elif int(imu_t.shape[1]) > seq_len:
            imu_t = imu_t[:, :seq_len, :]
        return imu_t

    def _imu_context(self, imu_seq: torch.Tensor) -> torch.Tensor:
        if self.imu_dim <= 0:
            return torch.zeros((imu_seq.shape[0], self.hidden_dim), device=imu_seq.device, dtype=imu_seq.dtype)
        imu_mean = imu_seq.mean(dim=1)
        imu_rms = torch.sqrt(torch.mean(torch.square(imu_seq), dim=1) + 1e-06)
        imu_seg = torch.cat([imu_mean, imu_rms], dim=-1)
        return self.imu_context_proj(imu_seg)

    def _single_frame_perceptual_encode(self, spatial_feat: torch.Tensor) -> torch.Tensor:
        if spatial_feat.ndim != 3:
            raise ValueError('spatial_feat must be [B, T, H].')
        return self.perception_frame_mlp(spatial_feat)

    def _short_horizon_perceptual_encode(self, spatial_feat: torch.Tensor, context: torch.Tensor, imu_seq: torch.Tensor) -> torch.Tensor:
        return self._single_frame_perceptual_encode(spatial_feat)

    def _legacy_short_horizon_perceptual_encode(self, spatial_feat: torch.Tensor, context: torch.Tensor, imu_seq: torch.Tensor) -> torch.Tensor:
        if spatial_feat.ndim != 3:
            raise ValueError('spatial_feat must be [B, T, H].')
        bsz, seq_len, hid = spatial_feat.shape
        imu_step = self.imu_step_proj(imu_seq) if self.imu_dim > 0 else torch.zeros_like(spatial_feat)
        feat_t = (spatial_feat + imu_step).transpose(1, 2)
        feat_pad = F.pad(feat_t, (self.local_window - 1, 0))
        windows = feat_pad.unfold(dimension=2, size=self.local_window, step=1)
        windows = windows.permute(0, 2, 1, 3).reshape(bsz * seq_len, hid, self.local_window)
        gamma, beta = torch.chunk(self.film_mlp(context), chunks=2, dim=-1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)
        gamma_seq = gamma[:, None, :].expand(bsz, seq_len, hid).reshape(bsz * seq_len, hid)
        beta_seq = beta[:, None, :].expand(bsz, seq_len, hid).reshape(bsz * seq_len, hid)
        local = windows * gamma_seq.unsqueeze(-1) + beta_seq.unsqueeze(-1)
        for block in self.tcn_blocks:
            local = block(local)
        h_now = local[:, :, -1].reshape(bsz, seq_len, hid)
        return h_now * gamma.unsqueeze(1) + beta.unsqueeze(1)

    def _causal_local_encode(self, spatial_feat: torch.Tensor, context: torch.Tensor, imu_seq: torch.Tensor) -> torch.Tensor:
        return self._single_frame_perceptual_encode(spatial_feat)

    def encode(self, x: torch.Tensor, imu: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        spatial_feat, enc_in = self._spatial_features(x)
        imu_seq = self._coerce_imu_seq(imu=imu, batch_size=int(x.shape[0]), seq_len=int(x.shape[1]), device=x.device, dtype=x.dtype)
        s_perc_seq = self._single_frame_perceptual_encode(spatial_feat)
        intent_feature_seq = self.intent_projection(s_perc_seq)
        z_drive_seq = F.softplus(self.drive_head(intent_feature_seq))
        z_grip_seq = F.softplus(self.grip_head(intent_feature_seq))
        return {'z_drive_seq': z_drive_seq, 'z_grip_seq': z_grip_seq, 'z_drive': z_drive_seq.mean(dim=1), 'z_grip': z_grip_seq.mean(dim=1), 'z_drive_curr': z_drive_seq[:, -1], 'z_grip_curr': z_grip_seq[:, -1], 's_perc_seq': s_perc_seq, 's_perc': s_perc_seq.mean(dim=1), 's_perc_curr': s_perc_seq[:, -1], 'intent_feature_seq': intent_feature_seq, 'intent_feature': intent_feature_seq.mean(dim=1), 'intent_feature_curr': intent_feature_seq[:, -1], 'state_seq': s_perc_seq, 'state_curr': s_perc_seq[:, -1], 'context_seg': torch.zeros((int(x.shape[0]), self.hidden_dim), device=x.device, dtype=x.dtype), 'imu_seq': imu_seq}

    def _coerce_person_id(self, person_id: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if person_id is None:
            return torch.zeros((batch_size,), device=device, dtype=torch.long)
        person_t = torch.as_tensor(person_id, device=device, dtype=torch.long).reshape(-1)
        if int(person_t.numel()) == 1 and batch_size > 1:
            person_t = person_t.expand(batch_size)
        if int(person_t.numel()) != int(batch_size):
            raise ValueError('person_id must align with batch size.')
        return torch.clamp(person_t, min=0, max=int(self.n_persons - 1))

    def _build_delayed_intention_buffer(self, z_seq: torch.Tensor, frame_valid: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if z_seq.ndim != 3:
            raise ValueError('z_seq must have shape [B, T, D].')
        batch_size = int(z_seq.shape[0])
        seq_len = int(z_seq.shape[1])
        if frame_valid is None:
            frame_valid = _stage1_encoder_frame_valid_mask_torch(seq_len, self.local_window, z_seq.device)
            frame_valid = frame_valid.unsqueeze(0).expand(batch_size, seq_len)
        else:
            frame_valid = torch.as_tensor(frame_valid, device=z_seq.device, dtype=torch.bool)
            if frame_valid.ndim == 1:
                frame_valid = frame_valid.unsqueeze(0).expand(batch_size, seq_len)
            if frame_valid.shape != z_seq.shape[:2]:
                raise ValueError('frame_valid must align with z_seq on [B, T].')
        z_pad = F.pad(z_seq, (0, 0, self.delay_kmax, 0))
        valid_pad = F.pad(frame_valid.to(dtype=torch.bool), (self.delay_kmax, 0), value=False)
        delayed_list: List[torch.Tensor] = []
        valid_list: List[torch.Tensor] = []
        for k in range(self.delay_kmin, self.delay_kmax + 1):
            start = int(self.delay_kmax - k)
            stop = int(start + seq_len)
            delayed_list.append(z_pad[:, start:stop, :])
            valid_list.append(valid_pad[:, start:stop])
        delayed_intents = torch.stack(delayed_list, dim=2)
        delayed_valid = torch.stack(valid_list, dim=-1)
        return (delayed_intents, delayed_valid)

    def aggregate_delayed_intents(self, z_drive_seq: torch.Tensor, z_grip_seq: torch.Tensor, person_id: Optional[torch.Tensor]=None, level_id: Optional[torch.Tensor]=None, frame_valid: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        if z_drive_seq.ndim != 3 or z_grip_seq.ndim != 3 or z_drive_seq.shape[:2] != z_grip_seq.shape[:2]:
            raise ValueError('z_drive_seq and z_grip_seq must both have shape [B, T, D] with matching [B, T].')
        person_id_t = self._coerce_person_id(person_id=person_id, batch_size=int(z_drive_seq.shape[0]), device=z_drive_seq.device)
        drive_buffer, drive_valid = self._build_delayed_intention_buffer(z_drive_seq, frame_valid=frame_valid)
        grip_buffer, grip_valid = self._build_delayed_intention_buffer(z_grip_seq, frame_valid=frame_valid)
        attn = self.person_temporal_attention(person_id_t, level_id=level_id)
        drive_weights = attn['drive_weights']
        grip_weights = attn['grip_weights']
        z_drive_eff = torch.sum(drive_buffer * drive_weights[:, None, :, None], dim=2)
        z_grip_eff = torch.sum(grip_buffer * grip_weights[:, None, :, None], dim=2)
        intent_window_valid = torch.all(drive_valid, dim=-1) & torch.all(grip_valid, dim=-1)
        return {'person_id': person_id_t, 'level_id': attn['level_id'], 'drive_buffer': drive_buffer, 'grip_buffer': grip_buffer, 'drive_buffer_valid': drive_valid, 'grip_buffer_valid': grip_valid, 'intent_window_valid': intent_window_valid, 'shared_attention_logits': attn['logits'], 'shared_attention_weights': attn['weights'], 'drive_attention_logits': attn['drive_logits'], 'grip_attention_logits': attn['grip_logits'], 'drive_attention_weights': drive_weights, 'grip_attention_weights': grip_weights, 'z_drive_effective_seq': z_drive_eff, 'z_grip_effective_seq': z_grip_eff}

    def _decode_biomimetic_dynamics(self, pose_curr: torch.Tensor, velocity_curr: torch.Tensor, z_drive_eff: torch.Tensor, z_grip_eff: torch.Tensor, imu_curr: Optional[torch.Tensor]=None, dt: float=0.01) -> Dict[str, torch.Tensor]:
        return self.biomimetic_impedance_decoder(z_drive_eff=z_drive_eff, z_grip_eff=z_grip_eff, pose_curr=pose_curr, velocity_curr=velocity_curr, imu_curr=imu_curr, drive_gain=self.B_gain, downward_drift=self.downward_drift, dt=dt)

    def reconstruct_perception_state(self, s_perc_seq: torch.Tensor) -> torch.Tensor:
        if s_perc_seq.ndim != 3:
            raise ValueError('s_perc_seq must have shape [B, T, H].')
        return self.perception_recon_head(s_perc_seq)

    def get_synergy_basis(self) -> torch.Tensor:
        basis = F.softplus(self.synergy_basis_raw)
        return basis / basis.sum(dim=1, keepdim=True).clamp_min(1e-06)

    def get_grip_basis(self) -> torch.Tensor:
        basis = F.softplus(self.grip_basis_raw)
        return basis / basis.sum(dim=1, keepdim=True).clamp_min(1e-06)

    def drive_dictionary_decode(self, z_drive_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_seq = torch.clamp(z_drive_seq, min=0.0)
        basis = self.get_synergy_basis()
        drive_force_seq = torch.einsum('btd,dn->btn', alpha_seq, basis)
        return (alpha_seq, drive_force_seq)

    def grip_basis_decode(self, z_grip_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta_seq = torch.clamp(z_grip_seq, min=0.0)
        basis = self.get_grip_basis()
        grip_force_seq = torch.einsum('btg,gn->btn', beta_seq, basis)
        return (beta_seq, grip_force_seq)

    def decode_components(self, z_drive_seq: torch.Tensor, z_grip_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        alpha_seq, drive_force_seq = self.drive_dictionary_decode(z_drive_seq)
        beta_seq, grip_force_seq = self.grip_basis_decode(z_grip_seq)
        recon = drive_force_seq + grip_force_seq
        return {'alpha_seq': alpha_seq, 'beta_seq': beta_seq, 'synergy_basis': self.get_synergy_basis(), 'grip_basis': self.get_grip_basis(), 'drive_force_seq': drive_force_seq, 'grip_force_seq': grip_force_seq, 'recon': recon}

    def decode(self, z_drive_seq: torch.Tensor, z_grip_seq: torch.Tensor) -> torch.Tensor:
        return self.decode_components(z_drive_seq, z_grip_seq)['recon']

    def predict_grip_stiffness_logits(self, pose_seq: torch.Tensor, z_grip_seq: torch.Tensor, z_drive_seq: Optional[torch.Tensor]=None, imu_seq: Optional[torch.Tensor]=None, person_id: Optional[torch.Tensor]=None, frame_valid: Optional[torch.Tensor]=None) -> torch.Tensor:
        if z_drive_seq is None:
            raise ValueError('z_drive_seq is required for strict delayed grip decoding.')
        velocity_stub = torch.zeros_like(pose_seq)
        outputs = self.predict_next_pose_velocity(pose_seq=pose_seq, velocity_seq=velocity_stub, z_drive_seq=z_drive_seq, z_grip_seq=z_grip_seq, imu_seq=imu_seq, dt=1.0, person_id=person_id, frame_valid=frame_valid)
        return outputs['k_grip_logits']

    def predict_grip_stiffness(self, pose_seq: torch.Tensor, z_grip_seq: torch.Tensor, z_drive_seq: Optional[torch.Tensor]=None, imu_seq: Optional[torch.Tensor]=None, person_id: Optional[torch.Tensor]=None, frame_valid: Optional[torch.Tensor]=None) -> torch.Tensor:
        if z_drive_seq is None:
            raise ValueError('z_drive_seq is required for strict delayed grip decoding.')
        velocity_stub = torch.zeros_like(pose_seq)
        outputs = self.predict_next_pose_velocity(pose_seq=pose_seq, z_grip_seq=z_grip_seq, z_drive_seq=z_drive_seq, velocity_seq=velocity_stub, imu_seq=imu_seq, dt=1.0, person_id=person_id, frame_valid=frame_valid)
        return outputs['impedance_stiffness']

    def predict_velocity_delta_rate(self, pose_seq: torch.Tensor, velocity_seq: torch.Tensor, z_drive_seq: torch.Tensor, z_grip_seq: Optional[torch.Tensor]=None, imu_seq: Optional[torch.Tensor]=None, person_id: Optional[torch.Tensor]=None, frame_valid: Optional[torch.Tensor]=None, dt: float=0.01) -> torch.Tensor:
        if z_grip_seq is None:
            z_grip_seq = torch.zeros((int(z_drive_seq.shape[0]), int(z_drive_seq.shape[1]), self.z_grip_dim), device=z_drive_seq.device, dtype=z_drive_seq.dtype)
        outputs = self.predict_next_pose_velocity(pose_seq=pose_seq, velocity_seq=velocity_seq, z_drive_seq=z_drive_seq, z_grip_seq=z_grip_seq, imu_seq=imu_seq, dt=dt, person_id=person_id, frame_valid=frame_valid)
        return outputs['acc_pred']

    def predict_drive_torque_intent(self, pose_seq: torch.Tensor, velocity_seq: torch.Tensor, z_drive_seq: torch.Tensor, z_grip_seq: Optional[torch.Tensor]=None, imu_seq: Optional[torch.Tensor]=None, person_id: Optional[torch.Tensor]=None, frame_valid: Optional[torch.Tensor]=None) -> torch.Tensor:
        outputs = self.predict_next_pose_velocity(pose_seq=pose_seq, velocity_seq=velocity_seq, z_drive_seq=z_drive_seq, z_grip_seq=z_grip_seq if z_grip_seq is not None else torch.zeros((int(z_drive_seq.shape[0]), int(z_drive_seq.shape[1]), self.z_grip_dim), device=z_drive_seq.device, dtype=z_drive_seq.dtype), imu_seq=imu_seq, dt=0.01, person_id=person_id, frame_valid=frame_valid)
        return outputs['tau_drive']

    def predict_next_pose_velocity_from_current(self, pose_curr: torch.Tensor, velocity_curr: torch.Tensor, z_drive_curr: torch.Tensor, z_grip_curr: torch.Tensor, imu_curr: Optional[torch.Tensor]=None, dt: float=0.01) -> Dict[str, torch.Tensor]:
        if pose_curr.ndim != 3 or velocity_curr.ndim != 3 or z_drive_curr.ndim != 3 or (z_grip_curr.ndim != 3):
            raise ValueError('current-step tensors must all have shape [B, T, D].')
        return self._decode_biomimetic_dynamics(pose_curr=pose_curr, velocity_curr=velocity_curr, z_drive_eff=z_drive_curr, z_grip_eff=z_grip_curr, imu_curr=imu_curr, dt=dt)

    def predict_next_pose_velocity(self, pose_seq: torch.Tensor, velocity_seq: torch.Tensor, z_drive_seq: torch.Tensor, z_grip_seq: torch.Tensor, imu_seq: Optional[torch.Tensor]=None, dt: float=0.01, person_id: Optional[torch.Tensor]=None, level_id: Optional[torch.Tensor]=None, frame_valid: Optional[torch.Tensor]=None, torque_target_center: Optional[torch.Tensor]=None, torque_target_half_range: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        if pose_seq.ndim != 3 or int(pose_seq.shape[-1]) != self.pose_state_dim:
            raise ValueError(f'pose_seq must have shape [B, T, {self.pose_state_dim}].')
        if velocity_seq.ndim != 3 or int(velocity_seq.shape[-1]) != self.pose_state_dim:
            raise ValueError(f'velocity_seq must have shape [B, T, {self.pose_state_dim}].')
        if z_drive_seq.ndim != 3 or int(z_drive_seq.shape[-1]) != self.z_drive_dim:
            raise ValueError(f'z_drive_seq must have shape [B, T, {self.z_drive_dim}].')
        if z_grip_seq.ndim != 3 or int(z_grip_seq.shape[-1]) != self.z_grip_dim:
            raise ValueError(f'z_grip_seq must have shape [B, T, {self.z_grip_dim}].')
        if pose_seq.shape[:2] != velocity_seq.shape[:2] or pose_seq.shape[:2] != z_drive_seq.shape[:2] or pose_seq.shape[:2] != z_grip_seq.shape[:2]:
            raise ValueError('pose/velocity/latent sequences must align on [B, T].')
        batch_size = int(pose_seq.shape[0])
        seq_len = int(pose_seq.shape[1])
        if seq_len <= 1:
            empty_seq = pose_seq[:, :0, :].clone()
            empty_scalar = pose_seq.new_empty((batch_size, 0, self.grip_gate_dim))
            return {'k_grip': empty_scalar, 'k_grip_logits': empty_scalar, 'k_grip_broadcast': empty_seq, 'impedance_stiffness': empty_seq, 'impedance_damping': empty_seq, 'pose_curr': empty_seq, 'pose_free': empty_seq, 'pose_drift': empty_seq, 'pose_anchor': empty_seq, 'pose_next_pred': empty_seq, 'velocity_base_pred': empty_seq, 'velocity_free': empty_seq, 'tau_drive': pose_seq.new_empty((batch_size, 0, 3)), 'delta_v_drive': empty_seq, 'impedance_delta_v': empty_seq, 'drift_term': empty_seq, 'relax_term': empty_seq, 'delta_v_pred': empty_seq, 'acc_pred': empty_seq, 'velocity_next_pred': empty_seq, 'intent_window_valid': pose_seq.new_empty((batch_size, 0), dtype=torch.bool), 'shared_attention_logits': pose_seq.new_empty((batch_size, self.delay_window_len)), 'shared_attention_weights': pose_seq.new_empty((batch_size, self.delay_window_len)), 'drive_attention_logits': pose_seq.new_empty((batch_size, self.delay_window_len)), 'grip_attention_logits': pose_seq.new_empty((batch_size, self.delay_window_len)), 'drive_attention_weights': pose_seq.new_empty((batch_size, self.delay_window_len)), 'grip_attention_weights': pose_seq.new_empty((batch_size, self.delay_window_len)), 'z_drive_effective_seq': pose_seq.new_empty((batch_size, 0, self.z_drive_dim)), 'z_grip_effective_seq': pose_seq.new_empty((batch_size, 0, self.z_grip_dim)), 'drive_buffer': pose_seq.new_empty((batch_size, 0, self.delay_window_len, self.z_drive_dim)), 'grip_buffer': pose_seq.new_empty((batch_size, 0, self.delay_window_len, self.z_grip_dim))}
        if frame_valid is None:
            frame_valid = _stage1_encoder_frame_valid_mask_torch(seq_len, self.local_window, pose_seq.device)
            frame_valid = frame_valid.unsqueeze(0).expand(batch_size, seq_len)
        else:
            frame_valid = torch.as_tensor(frame_valid, device=pose_seq.device, dtype=torch.bool)
            if frame_valid.ndim == 1:
                frame_valid = frame_valid.unsqueeze(0).expand(batch_size, seq_len)
        aggregated = self.aggregate_delayed_intents(z_drive_seq=z_drive_seq, z_grip_seq=z_grip_seq, person_id=person_id, level_id=level_id, frame_valid=frame_valid)
        imu_curr = imu_seq[:, :-1, :] if imu_seq is not None and imu_seq.ndim == 3 else None
        decoded = self._decode_biomimetic_dynamics(pose_curr=pose_seq[:, :-1, :], velocity_curr=velocity_seq[:, :-1, :], z_drive_eff=aggregated['z_drive_effective_seq'][:, :-1, :], z_grip_eff=aggregated['z_grip_effective_seq'][:, :-1, :], imu_curr=imu_curr, dt=dt)
        if torque_target_center is not None and torque_target_half_range is not None:
            tau_norm = _apply_stage2_torque_minmax_scaler_torch(decoded['tau_drive'], torch.as_tensor(torque_target_center, device=decoded['tau_drive'].device, dtype=decoded['tau_drive'].dtype), torch.as_tensor(torque_target_half_range, device=decoded['tau_drive'].device, dtype=decoded['tau_drive'].dtype))
        else:
            tau_norm = decoded['tau_drive']
        decoded.update({'tau_pred_seq_normalized': tau_norm, 'torque_seq_normalized': tau_norm, 'tau_pred_seq': decoded['tau_drive'], 'tau_pred_seq_raw': decoded['tau_drive'], 'z_drive_effective_seq': aggregated['z_drive_effective_seq'][:, :-1, :], 'z_grip_effective_seq': aggregated['z_grip_effective_seq'][:, :-1, :], 'intent_window_valid': aggregated['intent_window_valid'][:, :-1], 'shared_attention_logits': aggregated['shared_attention_logits'], 'shared_attention_weights': aggregated['shared_attention_weights'], 'drive_attention_logits': aggregated['drive_attention_logits'], 'grip_attention_logits': aggregated['grip_attention_logits'], 'drive_attention_weights': aggregated['drive_attention_weights'], 'grip_attention_weights': aggregated['grip_attention_weights'], 'drive_buffer': aggregated['drive_buffer'][:, :-1, :, :], 'grip_buffer': aggregated['grip_buffer'][:, :-1, :, :], 'drive_buffer_valid': aggregated['drive_buffer_valid'][:, :-1, :], 'grip_buffer_valid': aggregated['grip_buffer_valid'][:, :-1, :], 'person_id': aggregated['person_id'], 'level_id': aggregated['level_id']})
        return decoded

def _apply_stage2_torque_minmax_scaler_torch(torque_seq: torch.Tensor, center: torch.Tensor, half_range: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
    arr = torch.as_tensor(torque_seq)
    if arr.ndim < 1 or int(arr.shape[-1]) != 3:
        raise ValueError('stage2 torque scaler expects the last dimension to be 3.')
    center_t = torch.as_tensor(center, device=arr.device, dtype=arr.dtype).reshape((1,) * max(0, arr.ndim - 1) + (-1,))
    half_range_t = torch.as_tensor(half_range, device=arr.device, dtype=arr.dtype).reshape((1,) * max(0, arr.ndim - 1) + (-1,))
    half_range_t = torch.clamp(half_range_t, min=float(max(eps, 1e-06)))
    return (arr - center_t) / half_range_t
GraphTemporalSplitAE = _GraphTemporalSplitAE
