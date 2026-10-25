"""generative_model: extracted algorithm components.

See README.txt and SOURCE_MANIFEST.json for scope and documented changes.
No study data, fitted weights or manuscript statistical workflow is included.
"""
from __future__ import annotations
import torch.nn.functional as F
from typing import Optional
from typing import Tuple
import math
import torch.nn as nn
import numpy as np
import torch
from tactile_components.organization.graph_blocks import GCNBlock, SAGEBlock, GATBlock

class GraphEncoder(nn.Module):

    def __init__(self, in_ch: int=1, hidden: int=128, spatial: str='gcn'):
        super().__init__()
        s = spatial.lower()
        if s == 'gcn':
            self.block = GCNBlock(in_ch, hidden)
        elif s in ('gnn', 'sage', 'sage-mean'):
            self.block = SAGEBlock(in_ch, hidden)
        elif s in ('gat', 'attention-gnn', 'attention'):
            self.block = GATBlock(in_ch, hidden, heads=4)
        else:
            raise ValueError(f'Unknown spatial: {spatial}')
        self.hidden = hidden

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        return self.block(x, A_hat)

class TemporalEncoder(nn.Module):

    def __init__(self, in_ch: int, latent: int=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_ch, hidden_size=latent, batch_first=True)

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(steps)
        return h[-1]

def _normalize_envelope_torch(env: torch.Tensor, norm: str='minmax', eps: float=1e-06) -> torch.Tensor:
    mode = str(norm or 'minmax').lower()
    if mode in ('none', 'raw'):
        return env
    if mode in ('mean', 'mean1'):
        denom = env.mean(dim=1, keepdim=True) + eps
        return env / denom
    if mode in ('z', 'zscore', 'standard'):
        mu = env.mean(dim=1, keepdim=True)
        sd = env.std(dim=1, keepdim=True, unbiased=False) + eps
        return (env - mu) / sd
    vmin = env.amin(dim=1, keepdim=True)
    vmax = env.amax(dim=1, keepdim=True)
    return (env - vmin) / (vmax - vmin + eps)

def _tactile_envelope_torch(x: torch.Tensor, mode: str='rms', norm: str='minmax', pow_exp: float=1.0, eps: float=1e-06) -> torch.Tensor:
    """Compute per-frame envelope from tactile sequence. x: (B,T,N) -> (B,T,1)."""
    m = str(mode or 'rms').lower()
    if m in ('abs', 'absmean', 'l1'):
        env = x.abs().mean(dim=-1, keepdim=True)
    elif m in ('std', 'stdev'):
        env = x.std(dim=-1, keepdim=True, unbiased=False)
    else:
        env = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    env = _normalize_envelope_torch(env, norm=norm, eps=eps)
    if pow_exp is not None and float(pow_exp) != 1.0:
        env = torch.clamp(env, min=0.0) ** float(pow_exp)
    return env

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -emb)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        out = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            out = F.pad(out, (0, 1))
        return out

class TemporalSeqEncoder(nn.Module):

    def __init__(self, in_ch: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_ch, hidden_size=hidden, batch_first=True)

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(steps)
        return out

class StyleEncoder(nn.Module):
    """Encode tactile + IMU into a style vector."""

    def __init__(self, n_nodes: int, spatial: str='gcn', hidden: int=128, latent: int=128, normalize: bool=True):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.latent = latent
        self.normalize = bool(normalize)
        self.enc_tac = GraphEncoder(1, hidden, spatial)
        self.tem_tac = TemporalEncoder(hidden, latent)
        self.tem_imu = TemporalEncoder(6, latent)
        self.proj = nn.Linear(latent, hidden)

    def forward(self, x_tac: torch.Tensor, x_imu: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        x = x_tac.unsqueeze(-1)
        H = self.enc_tac(x, A_hat)
        se_tac = self.tem_tac(H.mean(dim=2))
        se_imu = self.tem_imu(x_imu)
        z = self.proj(se_tac + se_imu)
        if self.normalize:
            z = F.normalize(z, dim=-1)
        return z

class _GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (-ctx.lambd * grad_output, None)

def grad_reverse(x: torch.Tensor, lambd: float=1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)

class ContentEncoder(nn.Module):
    """Encode tactile + IMU into a content vector."""

    def __init__(self, n_nodes: int, spatial: str='gcn', hidden: int=128, latent: int=128):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.latent = latent
        self.enc_tac = GraphEncoder(1, hidden, spatial)
        self.tem_tac = TemporalEncoder(hidden, latent)
        self.tem_imu = TemporalEncoder(6, latent)
        self.proj = nn.Linear(latent, hidden)

    def forward(self, x_tac: torch.Tensor, x_imu: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        x = x_tac.unsqueeze(-1)
        H = self.enc_tac(x, A_hat)
        c_tac = self.tem_tac(H.mean(dim=2))
        c_imu = self.tem_imu(x_imu)
        return self.proj(c_tac + c_imu)

class LatentDecoder(nn.Module):
    """Decode content+style vectors back to tactile + IMU sequences."""

    def __init__(self, n_nodes: int, hidden: int=128, latent_c: int=128, latent_s: int=128, use_env_cond: bool=False, env_dim: int=1, env_scale: float=1.0, output_activation: str='linear', tactile_out_hi: Optional[np.ndarray]=None, imu_out_abs: Optional[np.ndarray]=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.latent_c = latent_c
        self.latent_s = latent_s
        self.use_env_cond = bool(use_env_cond)
        self.env_scale = float(env_scale)
        self.output_activation = str(output_activation or 'linear').lower()
        self.fc = nn.Sequential(nn.Linear(latent_c + latent_s, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.env_mlp = None
        if self.use_env_cond:
            self.env_mlp = nn.Sequential(nn.Linear(int(env_dim), hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.out_tac = nn.Linear(hidden, n_nodes)
        self.out_imu = nn.Linear(hidden, 6)
        tac_hi = np.ones((int(n_nodes),), dtype=np.float32) if tactile_out_hi is None else np.asarray(tactile_out_hi, dtype=np.float32).reshape(-1)
        imu_hi = np.ones((6,), dtype=np.float32) if imu_out_abs is None else np.asarray(imu_out_abs, dtype=np.float32).reshape(-1)
        if tac_hi.size != int(n_nodes):
            tac_hi = np.ones((int(n_nodes),), dtype=np.float32)
        if imu_hi.size != 6:
            imu_hi = np.ones((6,), dtype=np.float32)
        tac_hi = np.clip(np.nan_to_num(tac_hi, nan=1.0, posinf=1000000.0, neginf=1.0), 1.0, 1000000.0)
        imu_hi = np.clip(np.nan_to_num(imu_hi, nan=1.0, posinf=1000000.0, neginf=1.0), 0.001, 1000000.0)
        self.register_buffer('tactile_out_hi', torch.from_numpy(tac_hi.astype(np.float32)))
        self.register_buffer('imu_out_abs', torch.from_numpy(imu_hi.astype(np.float32)))

    def forward(self, z_c: torch.Tensor, z_s: torch.Tensor, seq_len: int, env: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([z_c, z_s], dim=-1)
        h0 = self.fc(z)
        x = h0.unsqueeze(1).repeat(1, seq_len, 1)
        if self.use_env_cond and self.env_mlp is not None and (env is not None):
            if env.dim() == 2:
                env = env.unsqueeze(-1)
            env_feat = self.env_mlp(env)
            x = x + self.env_scale * env_feat
        h, _ = self.lstm(x)
        tac = self.out_tac(h)
        imu = self.out_imu(h)
        if self.output_activation in ('bounded', 'sigmoid_tanh', 'stable'):
            tac_hi = self.tactile_out_hi.view(1, 1, -1)
            imu_hi = self.imu_out_abs.view(1, 1, -1)
            tac = torch.sigmoid(tac) * tac_hi
            imu = torch.tanh(imu) * imu_hi
        return (tac, imu)

class LatentSkillAE(nn.Module):
    """Content/style disentangled autoencoder with level heads."""

    def __init__(self, n_nodes: int, spatial: str='gcn', hidden: int=128, latent_c: int=128, latent_s: int=128, n_levels: int=0, use_env_cond: bool=False, env_mode: str='rms', env_norm: str='minmax', env_pow: float=1.0, env_scale: float=1.0, output_activation: str='linear', tactile_out_hi: Optional[np.ndarray]=None, imu_out_abs: Optional[np.ndarray]=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.latent_c = latent_c
        self.latent_s = latent_s
        self.n_levels = int(max(0, n_levels))
        self.use_env_cond = bool(use_env_cond)
        self.env_mode = str(env_mode or 'rms')
        self.env_norm = str(env_norm or 'minmax')
        self.env_pow = float(env_pow)
        self.env_scale = float(env_scale)
        self.content = ContentEncoder(n_nodes, spatial, hidden, latent_c)
        self.style = StyleEncoder(n_nodes, spatial, hidden, latent_s, normalize=False)
        self.decoder = LatentDecoder(n_nodes, hidden, hidden, hidden, use_env_cond=self.use_env_cond, env_dim=1, env_scale=self.env_scale, output_activation=output_activation, tactile_out_hi=tactile_out_hi, imu_out_abs=imu_out_abs)
        self.level_head_style = nn.Linear(hidden, self.n_levels) if self.n_levels > 0 else None
        self.level_head_content = nn.Linear(hidden, self.n_levels) if self.n_levels > 0 else None

    def encode(self, x_tac: torch.Tensor, x_imu: torch.Tensor, A_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_c = self.content(x_tac, x_imu, A_hat)
        z_s = self.style(x_tac, x_imu, A_hat)
        return (z_c, z_s)

    def decode(self, z_c: torch.Tensor, z_s: torch.Tensor, seq_len: int, env: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(z_c, z_s, seq_len, env=env)

    def _build_env(self, x_tac: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_env_cond:
            return None
        return _tactile_envelope_torch(x_tac, mode=self.env_mode, norm=self.env_norm, pow_exp=self.env_pow)

    def forward(self, x_tac: torch.Tensor, x_imu: torch.Tensor, A_hat: torch.Tensor, adv_lambda: float=1.0):
        z_c, z_s = self.encode(x_tac, x_imu, A_hat)
        env = self._build_env(x_tac)
        x_hat, imu_hat = self.decode(z_c, z_s, x_tac.shape[1], env=env)
        logits_style = None
        logits_content = None
        if self.n_levels > 0:
            logits_style = self.level_head_style(z_s) if self.level_head_style is not None else None
            if self.level_head_content is not None:
                logits_content = self.level_head_content(grad_reverse(z_c, adv_lambda))
        return (x_hat, imu_hat, logits_style, logits_content, z_c, z_s)

class LatentDenoiser(nn.Module):
    """Conditional noise predictor for latent style deltas."""

    def __init__(self, latent_dim: int, cond_dim: int, hidden: int=256, time_dim: int=128, n_levels: int=0):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.hidden = hidden
        self.n_levels = int(max(0, n_levels))
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim + (self.n_levels + 1 if self.n_levels > 0 else 0), hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.in_mlp = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU())
        self.out_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim))

    def forward(self, zt: torch.Tensor, cond: torch.Tensor, level_onehot: Optional[torch.Tensor], level_scalar: Optional[torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        if self.n_levels > 0:
            if level_onehot is None:
                level_onehot = torch.zeros((cond.shape[0], self.n_levels), device=cond.device)
            if level_scalar is None:
                level_scalar = torch.zeros((cond.shape[0], 1), device=cond.device)
            if level_scalar.dim() == 1:
                level_scalar = level_scalar.view(-1, 1)
            level_feat = torch.cat([level_onehot.float(), level_scalar.float()], dim=-1)
            cond = torch.cat([cond, level_feat], dim=-1)
        h = self.in_mlp(zt) + self.time_mlp(t) + self.cond_mlp(cond)
        return self.out_mlp(h)

class LatentDiffusion(nn.Module):

    def __init__(self, latent_dim: int, cond_dim: int, hidden: int=256, n_levels: int=0, timesteps: int=1000, beta_start: float=0.0001, beta_end: float=0.02):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.hidden = hidden
        self.timesteps = timesteps
        self.denoiser = LatentDenoiser(latent_dim=latent_dim, cond_dim=cond_dim, hidden=hidden, time_dim=hidden, n_levels=n_levels)
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.ones(1), alphas_bar[:-1]], dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('alphas_bar_prev', alphas_bar_prev)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1.0 - alphas_bar))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        self.register_buffer('posterior_var', posterior_var)
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar))

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = _extract_coeff(self.sqrt_alphas_bar, t, z0.shape)
        sqrt_om = _extract_coeff(self.sqrt_one_minus_alphas_bar, t, z0.shape)
        return sqrt_ab * z0 + sqrt_om * noise

    def predict_x0(self, zt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = _extract_coeff(self.sqrt_alphas_bar, t, zt.shape)
        sqrt_om = _extract_coeff(self.sqrt_one_minus_alphas_bar, t, zt.shape)
        return (zt - sqrt_om * eps) / sqrt_ab

    @torch.no_grad()
    def sample_delta(self, z_src: torch.Tensor, cond: torch.Tensor, level_onehot: Optional[torch.Tensor], level_scalar: Optional[torch.Tensor]) -> torch.Tensor:
        B = z_src.shape[0]
        device = z_src.device
        zt = torch.randn_like(z_src)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            eps = self.denoiser(zt, cond, level_onehot, level_scalar, t_batch)
            x0 = self.predict_x0(zt, t_batch, eps)
            if t == 0:
                zt = x0
                break
            coef1 = _extract_coeff(self.posterior_mean_coef1, t_batch, zt.shape)
            coef2 = _extract_coeff(self.posterior_mean_coef2, t_batch, zt.shape)
            mean = coef1 * x0 + coef2 * zt
            var = _extract_coeff(self.posterior_var, t_batch, zt.shape)
            noise = torch.randn_like(zt)
            zt = mean + torch.sqrt(var) * noise
        return zt

def _extract_coeff(coeff: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = coeff.gather(0, t)
    return out.view(-1, *[1] * (len(shape) - 1))
