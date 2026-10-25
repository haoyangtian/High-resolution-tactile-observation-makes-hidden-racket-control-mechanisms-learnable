"""linear_classifier: extracted algorithm components.

See README.txt and SOURCE_MANIFEST.json for scope and documented changes.
No study data, fitted weights or manuscript statistical workflow is included.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch

def _fit_linear_svm_hinge(x_train: np.ndarray, y_train: np.ndarray, *, c: float=1.0, lr: float=0.05, n_steps: int=600) -> Tuple[np.ndarray, float]:
    x_np = np.asarray(x_train, dtype=np.float32)
    y_np = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if x_np.ndim != 2 or x_np.shape[0] <= 0 or x_np.shape[0] != y_np.shape[0]:
        raise ValueError('linear svm requires x_train=[N,D], y_train=[N].')
    xt = torch.from_numpy(x_np)
    yt = torch.from_numpy(np.where(y_np > 0, 1.0, -1.0).astype(np.float32, copy=False))
    w = torch.zeros((int(x_np.shape[1]),), dtype=torch.float32, requires_grad=True)
    b = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([w, b], lr=float(max(lr, 0.0001)))
    c_val = float(max(c, 1e-06))
    for _step in range(int(max(50, n_steps))):
        optimizer.zero_grad(set_to_none=True)
        margin = yt * (xt @ w + b)
        hinge = torch.relu(1.0 - margin)
        loss = 0.5 * torch.sum(w * w) + c_val * torch.mean(hinge)
        loss.backward()
        optimizer.step()
    return (w.detach().cpu().numpy().astype(np.float64, copy=False), float(b.detach().cpu()))
