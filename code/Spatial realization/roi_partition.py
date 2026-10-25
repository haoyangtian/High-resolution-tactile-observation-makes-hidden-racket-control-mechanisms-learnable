"""roi_partition: extracted algorithm components.

See README.txt and SOURCE_MANIFEST.json for scope and documented changes.
No study data, fitted weights or manuscript statistical workflow is included.
"""
from __future__ import annotations
from typing import List
import math
import numpy as np

def _edges_from_adj(A) -> List[tuple]:
    if A is None:
        return []
    if hasattr(A, 'detach'):
        A = A.detach().cpu().numpy()
    A = np.asarray(A)
    if A.size == 0:
        return []
    A = A > 0
    edges = np.transpose(np.nonzero(np.triu(A, 1)))
    return [(int(i), int(j)) for i, j in edges]

def _edge_weights_from_embeddings(z: np.ndarray, edges: List[tuple], sigma: float):
    if z is None or not edges:
        return ([], 1.0)
    dists = []
    for u, v in edges:
        dists.append(float(np.linalg.norm(z[u] - z[v])))
    if not dists:
        return ([1.0 for _ in edges], 1.0)
    if sigma is None or sigma <= 0:
        sigma = float(np.median(dists)) if np.median(dists) > 1e-06 else 1.0
    sigma = max(sigma, 1e-06)
    weights = [float(math.exp(-(d * d) / (2.0 * sigma * sigma))) for d in dists]
    return (weights, sigma)

def _modularity_from_edges(labels: np.ndarray, edges: List[tuple], weights: List[float], n_nodes: int) -> float:
    if n_nodes <= 0 or not edges:
        return 0.0
    lab = np.asarray(labels, dtype=np.int64)
    k = np.zeros(n_nodes, dtype=np.float64)
    m = 0.0
    for (u, v), w in zip(edges, weights):
        k[u] += w
        k[v] += w
        m += w
    if m <= 1e-12:
        return 0.0
    q = 0.0
    for (u, v), w in zip(edges, weights):
        if lab[u] == lab[v]:
            q += w - k[u] * k[v] / (2.0 * m)
    return float(q / (2.0 * m))

def _partition_graph(z, edges, weights, resolution, seed, *, backend='leiden'):
    """Partition a weighted graph with an explicit, never silently changed backend.

    Leiden requires the optional igraph and leidenalg packages. Louvain is an
    explicitly selected alternative, not a substitute for Leiden experiment
    results. The returned objective is an algorithm output, not a significance
    test or a manuscript group comparison.
    """
    z = np.asarray(z)
    if z.ndim != 2 or not np.all(np.isfinite(z)):
        raise ValueError('z must be a finite node-by-feature matrix')
    n = len(z)
    edges = list(edges)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if len(edges) != len(weights) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError('Every edge needs one finite nonnegative weight')
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('resolution must be finite and positive')
    for edge in edges:
        if len(edge) != 2 or any((int(v) != v or v < 0 or v >= n for v in edge)):
            raise ValueError('Edge indices must identify existing nodes')
    if backend not in {'leiden', 'louvain'}:
        raise ValueError("backend must be 'leiden' or 'louvain'")
    if n == 0:
        return (np.empty(0, dtype=np.int64), 0.0)
    if not edges or not np.any(weights > 0):
        return (np.arange(n, dtype=np.int64), 0.0)
    if backend == 'leiden':
        try:
            import igraph as ig
            import leidenalg
        except ImportError as exc:
            raise ImportError('Leiden requires the optional igraph and leidenalg dependencies') from exc
        graph = ig.Graph(n=n, edges=edges, directed=False)
        graph.es['weight'] = weights.tolist()
        partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights=graph.es['weight'], resolution_parameter=float(resolution), seed=seed)
        return (np.asarray(partition.membership, dtype=np.int64), float(partition.modularity))
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(((int(u), int(v), float(w)) for (u, v), w in zip(edges, weights)))
    groups = louvain_communities(graph, weight='weight', resolution=float(resolution), seed=seed)
    labels = np.empty(n, dtype=np.int64)
    for label, nodes in enumerate(groups):
        labels[list(nodes)] = label
    return (labels, float(nx.algorithms.community.quality.modularity(graph, groups, weight='weight')))

def _reindex_labels_contiguous(labels: np.ndarray) -> np.ndarray:
    lab = np.asarray(labels, dtype=np.int64).ravel()
    if lab.size == 0:
        return lab
    uniq = {}
    out = np.zeros(lab.shape[0], dtype=np.int64)
    next_id = 0
    for i in range(lab.shape[0]):
        k = int(lab[i])
        if k not in uniq:
            uniq[k] = next_id
            next_id += 1
        out[i] = uniq[k]
    return out
