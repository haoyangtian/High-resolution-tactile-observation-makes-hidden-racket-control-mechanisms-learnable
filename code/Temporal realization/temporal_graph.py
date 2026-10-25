"""temporal_graph: extracted algorithm components.

See README.txt and SOURCE_MANIFEST.json for scope and documented changes.
No study data, fitted weights or manuscript statistical workflow is included.
"""
from __future__ import annotations
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import math
import numpy as np

def _te_roi_distance_tier_masks(roi_centers: Optional[np.ndarray], n_groups: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, object]]:
    n = int(max(0, int(n_groups)))
    if n <= 1:
        return (None, None, None, {'enabled': False, 'nearest_k': 0, 'valid_center_count': 0, 'reason': 'too_few_groups'})
    try:
        centers = np.asarray(roi_centers, dtype=np.float32)
    except Exception:
        centers = np.zeros((0, 2), dtype=np.float32)
    if centers.ndim != 2 or int(centers.shape[0]) < n or int(centers.shape[1]) < 2:
        return (None, None, None, {'enabled': False, 'nearest_k': 0, 'valid_center_count': 0, 'reason': 'missing_roi_centers'})
    centers = np.asarray(centers[:n, :2], dtype=np.float32)
    valid = np.all(np.isfinite(centers), axis=1)
    valid_count = int(np.sum(valid))
    if valid_count <= 1:
        return (None, None, None, {'enabled': False, 'nearest_k': 0, 'valid_center_count': int(valid_count), 'reason': 'invalid_roi_centers'})
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32)).astype(np.float32, copy=False)
    invalid_pair = ~valid[:, None] | ~valid[None, :]
    dist[invalid_pair] = np.inf
    np.fill_diagonal(dist, np.inf)
    nearest = np.zeros((n, n), dtype=bool)
    tiers = np.zeros((n, n), dtype=np.int16)
    nearest_k = int(max(1, math.floor(float(max(0, n - 1)) / 3.0)))
    mid_k = int(max(0, math.floor(float(max(0, n - 1)) / 3.0)))
    for src in range(n):
        if not bool(valid[src]):
            continue
        order = [int(j) for j in sorted(range(n), key=lambda j: (float(dist[src, j]) if np.isfinite(float(dist[src, j])) else float('inf'), int(j))) if int(j) != int(src) and np.isfinite(float(dist[src, j]))]
        if not order:
            continue
        k1 = int(min(len(order), nearest_k))
        k2 = int(min(len(order), k1 + mid_k))
        for dst in order[:k1]:
            nearest[src, int(dst)] = True
            tiers[src, int(dst)] = 1
        for dst in order[k1:k2]:
            tiers[src, int(dst)] = 2
        for dst in order[k2:]:
            tiers[src, int(dst)] = 3
    return (dist, nearest, tiers, {'enabled': True, 'nearest_k': int(nearest_k), 'valid_center_count': int(valid_count), 'reason': ''})

def _te_spatial_nearest_edge_allowed(src: int, dst: int, nearest_mask: Optional[np.ndarray], mode: str='either') -> bool:
    if nearest_mask is None:
        return True
    try:
        mask = np.asarray(nearest_mask, dtype=bool)
        n = int(mask.shape[0])
        s = int(src)
        d = int(dst)
        if s < 0 or d < 0 or s >= n or (d >= n) or (s == d):
            return False
        source_near = bool(mask[s, d])
        dest_near = bool(mask[d, s])
        mode_use = str(mode or 'either').strip().lower()
        if mode_use in {'source', 'src', 'out', 'forward'}:
            return bool(source_near)
        if mode_use in {'dest', 'dst', 'target', 'current', 'in'}:
            return bool(dest_near)
        if mode_use in {'mutual', 'both'}:
            return bool(source_near and dest_near)
        return bool(source_near or dest_near)
    except Exception:
        return True

def _te_join_int_list_field(values: Sequence[int]) -> str:
    vals = sorted({int(v) for v in list(values or []) if int(v) >= 0})
    return ';'.join((str(int(v)) for v in vals))

def _te_build_local_propagation_forest_from_te(te_weights: np.ndarray, lag_frames: np.ndarray, group_labels: Sequence[str], groups_sorted: Sequence[int], roi_centers: Optional[np.ndarray], ms_per_frame: float, *, nearest_mode='either', max_parents_per_root=1) -> Tuple[List[dict], np.ndarray, Dict[str, object]]:
    if nearest_mode not in {'either', 'source', 'src', 'out', 'forward', 'dest', 'dst', 'target', 'current', 'in', 'mutual', 'both'}:
        raise ValueError('Unknown nearest_mode')
    if int(max_parents_per_root) != max_parents_per_root or max_parents_per_root < 1:
        raise ValueError('max_parents_per_root must be a positive integer')
    w_arr = np.asarray(te_weights, dtype=np.float32)
    n_groups = int(w_arr.shape[0]) if w_arr.ndim == 2 and w_arr.shape[0] == w_arr.shape[1] else 0
    if n_groups <= 0:
        return ([], np.full((0,), -1, dtype=np.int32), {'source_count': 0, 'edge_count': 0})
    lag_arr = np.asarray(lag_frames, dtype=np.int32)
    if lag_arr.shape != (n_groups, n_groups):
        lag_arr = np.zeros((n_groups, n_groups), dtype=np.int32)
    centers = np.asarray(roi_centers, dtype=np.float32) if roi_centers is not None else np.zeros((0, 2), dtype=np.float32)
    dist_mat, nearest_mask, tier_mat, tier_meta = _te_roi_distance_tier_masks(centers, int(n_groups))
    w_clean = np.where(np.isfinite(w_arr), np.maximum(w_arr, 0.0), 0.0).astype(np.float32)
    np.fill_diagonal(w_clean, 0.0)
    out_strength = np.sum(w_clean, axis=1)
    in_strength = np.sum(w_clean, axis=0)
    net_source_score = out_strength - in_strength
    order_nodes = [int(i) for i in sorted(range(n_groups), key=lambda idx: (-float(net_source_score[int(idx)]), -float(out_strength[int(idx)]), float(in_strength[int(idx)]), int(idx)))]
    rank_arr = np.full((n_groups,), -1, dtype=np.int32)
    for rank_i, node in enumerate(order_nodes):
        rank_arr[int(node)] = int(rank_i)
    parent_edges_by_dst: Dict[int, List[dict]] = {}
    roots_by_node: Dict[int, set] = {}
    comps_by_node: Dict[int, set] = {}
    root_to_comp: Dict[int, int] = {}

    def _ensure_source(node_i: int) -> set:
        node_i = int(node_i)
        if node_i not in roots_by_node:
            roots_by_node[node_i] = {node_i}
        if node_i not in root_to_comp:
            root_to_comp[node_i] = int(len(root_to_comp))
        comps_by_node[node_i] = {int(root_to_comp[node_i])}
        return set(roots_by_node[node_i])

    def _roots_for_node(node_i: int) -> set:
        node_i = int(node_i)
        if node_i in roots_by_node and roots_by_node[node_i]:
            return set(roots_by_node[node_i])
        return _ensure_source(node_i)

    def _has_path(src: int, dst: int) -> bool:
        src = int(src)
        dst = int(dst)
        stack = [int(src)]
        seen = set()
        while stack:
            cur = int(stack.pop())
            if cur == int(dst):
                return True
            if cur in seen:
                continue
            seen.add(cur)
            for edge_list in parent_edges_by_dst.values():
                for edge in edge_list:
                    if int(edge.get('src_idx', -1)) == cur:
                        nxt = int(edge.get('dst_idx', -1))
                        if nxt not in seen:
                            stack.append(nxt)
        return False
    selected_rows: List[dict] = []
    processed = set()
    for dst in order_nodes:
        dst = int(dst)
        candidates = []
        for src in processed:
            src = int(src)
            if src == dst:
                continue
            w_val = float(w_clean[src, dst])
            if not np.isfinite(w_val) or w_val <= 0.0:
                continue
            if not _te_spatial_nearest_edge_allowed(src, dst, nearest_mask, mode=nearest_mode):
                continue
            if _has_path(dst, src):
                continue
            dist_val = float(dist_mat[src, dst]) if dist_mat is not None and src < dist_mat.shape[0] and (dst < dist_mat.shape[1]) else float('nan')
            tier_src_to_dst = int(tier_mat[src, dst]) if tier_mat is not None and src < tier_mat.shape[0] and (dst < tier_mat.shape[1]) else 0
            tier_dst_to_src = int(tier_mat[dst, src]) if tier_mat is not None and dst < tier_mat.shape[0] and (src < tier_mat.shape[1]) else 0
            lag_i = int(lag_arr[src, dst]) if int(lag_arr[src, dst]) > 0 else 1
            candidates.append({'src': int(src), 'dst': int(dst), 'weight': float(w_val), 'lag_frames': int(lag_i), 'lag_ms': float(lag_i) * float(ms_per_frame), 'spatial_distance': float(dist_val), 'spatial_tier_src_to_dst': int(tier_src_to_dst), 'spatial_tier_dst_to_src': int(tier_dst_to_src), 'sort_key': (-float(w_val), float(dist_val) if np.isfinite(dist_val) else float('inf'), int(rank_arr[src]) if src < rank_arr.size else int(src), int(src))})
        candidates.sort(key=lambda x: x['sort_key'])
        selected_for_dst = []
        selected_count_by_root: Dict[int, int] = {}
        for cand in candidates:
            src = int(cand['src'])
            roots_src = sorted((int(v) for v in _roots_for_node(src) if int(v) >= 0))
            accepted_roots = []
            for root_i in roots_src:
                cnt_i = int(selected_count_by_root.get(int(root_i), 0))
                if cnt_i >= int(max_parents_per_root):
                    continue
                accepted_roots.append(int(root_i))
                selected_count_by_root[int(root_i)] = int(cnt_i + 1)
            if not accepted_roots:
                continue
            cand_use = dict(cand)
            cand_use['source_indices'] = sorted({int(v) for v in accepted_roots if int(v) >= 0})
            cand_use['component_ids'] = sorted({int(root_to_comp.setdefault(int(v), len(root_to_comp))) for v in accepted_roots if int(v) >= 0})
            selected_for_dst.append(cand_use)
        if selected_for_dst:
            roots_dst = set()
            comps_dst = set()
            for edge in selected_for_dst:
                src = int(edge['src'])
                dst_i = int(edge['dst'])
                roots_edge = set((int(v) for v in edge.get('source_indices', []) if int(v) >= 0))
                comps_edge = set((int(v) for v in edge.get('component_ids', []) if int(v) >= 0))
                roots_dst.update(roots_edge)
                comps_dst.update(comps_edge)
                row = {'src_idx': int(src), 'dst_idx': int(dst_i), 'src_group_id': int(groups_sorted[src]) if src < len(groups_sorted) else int(src), 'dst_group_id': int(groups_sorted[dst_i]) if dst_i < len(groups_sorted) else int(dst_i), 'src_name': str(group_labels[src]) if src < len(group_labels) else f'ROI {src + 1}', 'dst_name': str(group_labels[dst_i]) if dst_i < len(group_labels) else f'ROI {dst_i + 1}', 'rank_src_consensus': int(rank_arr[src]) if src < rank_arr.size else int(src), 'rank_dst_consensus': int(rank_arr[dst_i]) if dst_i < rank_arr.size else int(dst_i), 'precedence_prob': float('nan'), 'precedence_p_raw': float('nan'), 'precedence_q_fdr': float('nan'), 'delta_med_ms': float(edge.get('lag_ms', 0.0)), 'lag_boot_median_ms': float(edge.get('lag_ms', 0.0)), 'delta_low_ms': float(edge.get('lag_ms', 0.0)), 'delta_high_ms': float(edge.get('lag_ms', 0.0)), 'person_recurrence': float('nan'), 'direct_edge_support': float(edge.get('weight', 0.0)), 'weight': float(edge.get('weight', 0.0)), 'spatial_distance': float(edge.get('spatial_distance', float('nan'))), 'spatial_tier_src_to_dst': int(edge.get('spatial_tier_src_to_dst', 0)), 'spatial_tier_dst_to_src': int(edge.get('spatial_tier_dst_to_src', 0)), 'source_indices': list(sorted(roots_edge)), 'component_ids': list(sorted(comps_edge)), 'source_indices_text': _te_join_int_list_field(sorted(roots_edge)), 'component_ids_text': _te_join_int_list_field(sorted(comps_edge)), 'selection_mode': 'person_significant_nearest_te', 'stable_edge': True}
                selected_rows.append(row)
            roots_by_node[int(dst)] = set((int(v) for v in roots_dst if int(v) >= 0)) or {int(dst)}
            comps_by_node[int(dst)] = set((int(v) for v in comps_dst if int(v) >= 0))
        else:
            _ensure_source(int(dst))
        processed.add(int(dst))
    meta = {'source_count': int(len(root_to_comp)), 'edge_count': int(len(selected_rows)), 'nearest_mode': str(nearest_mode), 'max_parents_per_root': int(max_parents_per_root), 'nearest_meta': dict(tier_meta or {}), 'fallback_parent_enabled': False}
    return (selected_rows, rank_arr, meta)
build_temporal_forest = _te_build_local_propagation_forest_from_te
