"""graph_topology: extracted algorithm components.

See README.txt and SOURCE_MANIFEST.json for scope and documented changes.
No study data, fitted weights or manuscript statistical workflow is included.
"""
from __future__ import annotations
from sklearn.neighbors import NearestNeighbors
from typing import Optional
import numpy as np
import torch

def _adj_from_edges(n: int, edges, self_loop: bool=True) -> torch.Tensor:
    """Docstring."""
    A = np.zeros((n, n), dtype=np.float32)
    for i, j in edges:
        if i == j:
            continue
        A[i, j] = 1.0
        A[j, i] = 1.0
    if self_loop:
        np.fill_diagonal(A, 1.0)
    return torch.tensor(A, dtype=torch.float32)

def _pairwise_dists(coords: np.ndarray) -> np.ndarray:
    """Docstring."""
    X = np.asarray(coords, dtype=np.float32)
    G = X @ X.T
    sq = np.expand_dims(np.sum(X ** 2, axis=1), 1)
    D2 = np.maximum(sq + sq.T - 2 * G, 0.0)
    return np.sqrt(D2, out=D2)

def build_knn_adj(coords, k: int=8, self_loop: bool=True, mutual: bool=False) -> torch.Tensor:
    """
    Docstring.
    Docstring.
    """
    X = np.asarray(coords, dtype=np.float32)
    n = X.shape[0]
    if n == 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    if k < 0 or int(k) != k:
        raise ValueError('k must be a nonnegative integer')
    if n == 1 or k == 0:
        return _adj_from_edges(n, [], self_loop=self_loop)
    k_eff = min(k, max(1, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm='auto').fit(X)
    _, indices = nbrs.kneighbors(X)
    edges = set()
    if not mutual:
        for i in range(n):
            for j in indices[i][1:]:
                edges.add((min(i, j), max(i, j)))
    else:
        inv = [set() for _ in range(n)]
        for i in range(n):
            for j in indices[i][1:]:
                inv[i].add(j)
        for i in range(n):
            for j in indices[i][1:]:
                if i in inv[j]:
                    edges.add((min(i, j), max(i, j)))
    return _adj_from_edges(n, edges, self_loop=self_loop)

def build_radius_adj(coords, radius: Optional[float]=None, percentile: int=15, self_loop: bool=True) -> torch.Tensor:
    """
    Docstring.
    Docstring.
    """
    X = np.asarray(coords, dtype=np.float32)
    n = X.shape[0]
    if n == 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    D = _pairwise_dists(X)
    if radius is None:
        tri = D[np.triu_indices(n, k=1)]
        tri = tri[tri > 0]
        r = float(np.percentile(tri, percentile)) if len(tri) else 1.0
    else:
        r = float(radius)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= r:
                edges.add((i, j))
    return _adj_from_edges(n, edges, self_loop=self_loop)

def build_mst_adj(coords, self_loop: bool=True) -> torch.Tensor:
    """Docstring."""
    X = np.asarray(coords, dtype=np.float32)
    n = X.shape[0]
    if n <= 1:
        return _adj_from_edges(n, [], self_loop=self_loop)
    D = _pairwise_dists(X)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    edges = set()
    for _ in range(n - 1):
        best = None
        best_w = 1e+18
        for u in np.where(visited)[0]:
            dists = D[u]
            for v in np.where(~visited)[0]:
                w = dists[v]
                if w < best_w:
                    best_w = w
                    best = (u, v)
        u, v = best
        edges.add((min(u, v), max(u, v)))
        visited[v] = True
    return _adj_from_edges(n, edges, self_loop=self_loop)

def build_mst_knn_hybrid(coords, k: int=4, self_loop: bool=True) -> torch.Tensor:
    """Docstring."""
    A_mst = build_mst_adj(coords, self_loop=False).numpy()
    A_knn = build_knn_adj(coords, k=k, self_loop=False).numpy()
    A = (A_mst + A_knn > 0).astype(np.float32)
    if self_loop:
        np.fill_diagonal(A, 1.0)
    return torch.tensor(A, dtype=torch.float32)

def build_fully_connected(coords, self_loop: bool=True) -> torch.Tensor:
    """Docstring."""
    n = len(coords)
    A = np.ones((n, n), dtype=np.float32)
    if not self_loop:
        np.fill_diagonal(A, 0.0)
    return torch.tensor(A, dtype=torch.float32)
