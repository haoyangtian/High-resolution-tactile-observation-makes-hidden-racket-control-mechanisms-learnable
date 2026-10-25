"""grip_geometry: extracted algorithm components.

See README.txt and SOURCE_MANIFEST.json for scope and documented changes.
No study data, fitted weights or manuscript statistical workflow is included.
"""
from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class HandAxis:
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    perp_len: float

@dataclass
class OctagonPrism:
    center: np.ndarray
    axis_dir: np.ndarray
    half_len: float
    apothem: float

def _prism_basis(axis_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axis = axis_dir / (np.linalg.norm(axis_dir) + 1e-08)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    v = z_axis - axis * float(np.dot(z_axis, axis))
    if float(np.linalg.norm(v)) < 1e-06:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v = tmp - axis * float(np.dot(tmp, axis))
    v = v / (np.linalg.norm(v) + 1e-08)
    u = np.cross(axis, v)
    u = u / (np.linalg.norm(u) + 1e-08)
    return (u.astype(np.float32), v.astype(np.float32))

def _points_in_poly2d(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    poly = np.asarray(poly, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2 or poly.ndim != 2 or (poly.shape[0] < 3):
        return np.zeros((pts.shape[0],), dtype=bool)
    x = pts[:, 0]
    y = pts[:, 1]
    xp = poly[:, 0]
    yp = poly[:, 1]
    inside = np.zeros((pts.shape[0],), dtype=bool)
    j = int(poly.shape[0] - 1)
    for i in range(poly.shape[0]):
        xi, yi = (float(xp[i]), float(yp[i]))
        xj, yj = (float(xp[j]), float(yp[j]))
        cond = (yi > y) != (yj > y)
        denom = yj - yi
        denom = denom + 1e-12 if abs(denom) < 1e-12 else denom
        x_inters = (xj - xi) * (y - yi) / denom + xi
        hit = cond & (x < x_inters)
        inside ^= hit
        j = i
    return inside

def _closest_point_on_poly2d(point: np.ndarray, poly: np.ndarray) -> np.ndarray:
    pt = np.asarray(point, dtype=np.float32)
    poly = np.asarray(poly, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 2:
        return pt.copy()
    best = poly[0]
    best_d2 = float('inf')
    n = poly.shape[0]
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-08:
            proj = a
        else:
            t = float(np.dot(pt - a, ab)) / denom
            t = float(np.clip(t, 0.0, 1.0))
            proj = a + t * ab
        d2 = float(np.dot(pt - proj, pt - proj))
        if d2 < best_d2:
            best_d2 = d2
            best = proj
    return best.copy()

def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-08)
    x, y, z = (float(axis[0]), float(axis[1]), float(axis[2]))
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s], [z * x * C - y * s, z * y * C + x * s, c + z * z * C]], dtype=np.float32)

def _apply_rigid(points: np.ndarray, R: np.ndarray, center: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim == 1:
        return R @ (pts - center) + center + t
    return (R @ (pts - center).T).T + center[None, :] + t[None, :]

def _apply_rotation(vecs: np.ndarray, R: np.ndarray) -> np.ndarray:
    v = np.asarray(vecs, dtype=np.float32)
    if v.ndim == 1:
        return R @ v
    return (R @ v.T).T

def _fit_similarity_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate similarity transform (scale, rotation, translation) from src -> dst.
    Returns scale s, rotation R (3x3), translation t so that:
        dst ~= s * (src @ R.T) + t
    """
    src_arr = np.asarray(src, dtype=np.float32)
    dst_arr = np.asarray(dst, dtype=np.float32)
    if src_arr.ndim != 2 or dst_arr.ndim != 2 or src_arr.shape[0] != dst_arr.shape[0] or (src_arr.shape[1] < 3) or (dst_arr.shape[1] < 3) or (src_arr.shape[0] < 3):
        return (1.0, np.eye(3, dtype=np.float32), np.zeros((3,), dtype=np.float32))
    src3 = src_arr[:, :3].astype(np.float32)
    dst3 = dst_arr[:, :3].astype(np.float32)
    mu_src = src3.mean(axis=0)
    mu_dst = dst3.mean(axis=0)
    src_c = src3 - mu_src[None, :]
    dst_c = dst3 - mu_dst[None, :]
    var_src = float(np.sum(src_c * src_c))
    if not np.isfinite(var_src) or var_src < 1e-09:
        return (1.0, np.eye(3, dtype=np.float32), (mu_dst - mu_src).astype(np.float32))
    cov = src_c.T @ dst_c
    try:
        u, svals, vt = np.linalg.svd(cov.astype(np.float64), full_matrices=True)
    except Exception:
        return (1.0, np.eye(3, dtype=np.float32), (mu_dst - mu_src).astype(np.float32))
    R = (vt.T @ u.T).astype(np.float64)
    if float(np.linalg.det(R)) < 0.0:
        vt[-1, :] *= -1.0
        R = vt.T @ u.T
    scale = float(np.sum(svals) / (var_src + 1e-12))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    t = mu_dst.astype(np.float64) - scale * (R @ mu_src.astype(np.float64))
    return (float(scale), R.astype(np.float32), t.astype(np.float32))
