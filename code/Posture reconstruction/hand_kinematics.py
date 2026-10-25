"""hand_kinematics: extracted algorithm components.

See README.txt and SOURCE_MANIFEST.json for scope and documented changes.
No study data, fitted weights or manuscript statistical workflow is included.
"""
from __future__ import annotations
from typing import Tuple
import math
import numpy as np

class PlanarHandKinematics:

    @staticmethod
    def _forward_finger_3seg(L1: float, L2: float, L3: float, mcp_deg: float, pip_deg: float, dip_deg: float, which: str) -> Tuple[float, float]:
        """标准三节手指平面链条的中心点位置 (y, z)，在 finger 局部坐标系下。"""
        t1 = math.radians(mcp_deg)
        t2 = t1 + math.radians(pip_deg)
        t3 = t2 + math.radians(dip_deg)

        def seg_center(L, theta):
            return (L * 0.5 * math.cos(theta), L * 0.5 * math.sin(theta))
        y1_tip = L1 * math.cos(t1)
        z1_tip = L1 * math.sin(t1)
        y2_tip = y1_tip + L2 * math.cos(t2)
        z2_tip = z1_tip + L2 * math.sin(t2)
        if which == 'prox':
            return seg_center(L1, t1)
        elif which == 'mid':
            y_c2, z_c2 = seg_center(L2, t2)
            return (y1_tip + y_c2, z1_tip + z_c2)
        else:
            y_c3, z_c3 = seg_center(L3, t3)
            return (y2_tip + y_c3, z2_tip + z_c3)

    @staticmethod
    def _forward_finger_4seg(L74: float, L76: float, L65: float, L51: float, mcp_deg: float, pip_deg: float, dip_deg: float) -> Tuple[float, float, float, float, float, float, float, float]:
        mcp_half = 0.5 * float(mcp_deg)
        t0 = math.radians(mcp_half)
        t1 = t0 + math.radians(mcp_half)
        t2 = t1 + math.radians(pip_deg)
        t3 = t2 + math.radians(dip_deg)

        def step(L, theta):
            return (L * math.cos(theta), L * math.sin(theta))
        y7, z7 = step(L74, t0)
        y6, z6 = step(L76, t1)
        y6 += y7
        z6 += z7
        y5, z5 = step(L65, t2)
        y5 += y6
        z5 += z6
        y1, z1 = step(L51, t3)
        y1 += y5
        z1 += z5
        return (y7, z7, y6, z6, y5, z5, y1, z1)

    @staticmethod
    def _forward_thumb_2seg(L1: float, L2: float, a1_deg: float, a2_deg: float) -> Tuple[float, float, float, float]:
        t1 = math.radians(a1_deg)
        t2 = t1 + math.radians(a2_deg)
        y1 = L1 * math.cos(t1)
        z1 = L1 * math.sin(t1)
        y2 = y1 + L2 * math.cos(t2)
        z2 = z1 + L2 * math.sin(t2)
        return (y1, z1, y2, z2)

    @staticmethod
    def _rotate_vec_about_axis(vec: np.ndarray, axis_dir: np.ndarray, angle_rad: float) -> np.ndarray:
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-08)
        v = np.asarray(vec, dtype=np.float32)
        cos_t = float(np.cos(angle_rad))
        sin_t = float(np.sin(angle_rad))
        return v * cos_t + np.cross(axis_dir, v) * sin_t + axis_dir * float(np.dot(axis_dir, v)) * (1.0 - cos_t)

    @staticmethod
    def _rotate_point_about_axis(point: np.ndarray, axis_p0: np.ndarray, axis_dir: np.ndarray, angle_rad: float) -> np.ndarray:
        return axis_p0 + PlanarHandKinematics._rotate_vec_about_axis(point - axis_p0, axis_dir, angle_rad)
