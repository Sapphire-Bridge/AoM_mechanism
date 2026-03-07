from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from aom.features.effect_matrix import FeatureEffectMatrix
from aom.stats import bootstrap_ci


def _vector(matrix: FeatureEffectMatrix, fid: int) -> np.ndarray:
    vals: List[float] = []
    row = matrix.effects.get(int(fid), {})
    for cid in matrix.condition_ids:
        v = float(row.get(str(cid), float("nan")))
        vals.append(v)
    arr = np.asarray(vals, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= eps or nb <= eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cluster_feature_families(
    matrix: FeatureEffectMatrix,
    *,
    similarity_threshold: float = 0.8,
) -> Dict[int, List[int]]:
    """
    Deterministic greedy clustering by cosine similarity to cluster centroids.
    """
    if not matrix.feature_ids:
        return {}
    vectors = {int(fid): _vector(matrix, int(fid)) for fid in matrix.feature_ids}
    order = sorted(int(fid) for fid in matrix.feature_ids)

    clusters: Dict[int, List[int]] = {}
    centroids: Dict[int, np.ndarray] = {}
    next_cluster = 0
    for fid in order:
        v = vectors[int(fid)]
        best_cid = None
        best_sim = -1.0
        for cid, cvec in centroids.items():
            sim = _cosine(v, cvec)
            if sim > best_sim:
                best_sim = sim
                best_cid = int(cid)
        if best_cid is None or float(best_sim) < float(similarity_threshold):
            cid = int(next_cluster)
            next_cluster += 1
            clusters[cid] = [int(fid)]
            centroids[cid] = v.copy()
        else:
            cid = int(best_cid)
            clusters[cid].append(int(fid))
            mats = np.stack([vectors[x] for x in clusters[cid]], axis=0)
            centroids[cid] = mats.mean(axis=0)
    return clusters


def _labels_from_clusters(clusters: Mapping[int, Sequence[int]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for cid, fids in clusters.items():
        for fid in fids:
            out[int(fid)] = int(cid)
    return out


def adjusted_rand_index(labels_a: Mapping[int, int], labels_b: Mapping[int, int]) -> float:
    """
    ARI without sklearn dependency.
    """
    keys = sorted(set(int(k) for k in labels_a.keys()) & set(int(k) for k in labels_b.keys()))
    if not keys:
        return float("nan")
    a = [int(labels_a[k]) for k in keys]
    b = [int(labels_b[k]) for k in keys]
    n = len(keys)
    if n < 2:
        return 1.0

    # Contingency table.
    a_ids = sorted(set(a))
    b_ids = sorted(set(b))
    a_index = {lab: i for i, lab in enumerate(a_ids)}
    b_index = {lab: i for i, lab in enumerate(b_ids)}
    table = np.zeros((len(a_ids), len(b_ids)), dtype=int)
    for ai, bi in zip(a, b):
        table[a_index[ai], b_index[bi]] += 1

    def comb2(x: np.ndarray | int) -> float:
        xx = np.asarray(x, dtype=float)
        return float(np.sum(xx * (xx - 1.0) / 2.0))

    sum_nij = comb2(table)
    sum_ai = comb2(table.sum(axis=1))
    sum_bj = comb2(table.sum(axis=0))
    total = comb2(n)
    if total <= 0.0:
        return 1.0
    expected = (sum_ai * sum_bj) / total
    max_index = 0.5 * (sum_ai + sum_bj)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 1.0
    return float((sum_nij - expected) / denom)


@dataclass(frozen=True)
class ClusteringStability:
    mean_ari: float
    ari_ci_low: float
    ari_ci_high: float
    n_bootstrap: int


def bootstrap_stability(
    matrix: FeatureEffectMatrix,
    *,
    similarity_threshold: float = 0.8,
    n_bootstrap: int = 200,
    seed: int = 42,
    ci: float = 0.95,
) -> ClusteringStability:
    base = cluster_feature_families(matrix, similarity_threshold=float(similarity_threshold))
    base_labels = _labels_from_clusters(base)
    if not base_labels:
        return ClusteringStability(mean_ari=float("nan"), ari_ci_low=float("nan"), ari_ci_high=float("nan"), n_bootstrap=0)

    rng = np.random.RandomState(int(seed))
    conds = list(matrix.condition_ids)
    if not conds:
        return ClusteringStability(mean_ari=float("nan"), ari_ci_low=float("nan"), ari_ci_high=float("nan"), n_bootstrap=0)

    aris: List[float] = []
    for _ in range(int(n_bootstrap)):
        sample_idx = rng.randint(0, len(conds), size=len(conds))
        sampled_conds = [conds[i] for i in sample_idx]
        boot_effects: Dict[int, Dict[str, float]] = {}
        for fid in matrix.feature_ids:
            src = matrix.effects.get(int(fid), {})
            boot_effects[int(fid)] = {f"{j}": float(src.get(str(c), 0.0)) for j, c in enumerate(sampled_conds)}
        boot_matrix = FeatureEffectMatrix(
            feature_ids=tuple(matrix.feature_ids),
            condition_ids=tuple(str(i) for i in range(len(sampled_conds))),
            effects=boot_effects,
        )
        boot = cluster_feature_families(boot_matrix, similarity_threshold=float(similarity_threshold))
        ari = adjusted_rand_index(base_labels, _labels_from_clusters(boot))
        if np.isfinite(ari):
            aris.append(float(ari))

    mean_ari, lo, hi = bootstrap_ci(aris, n_bootstrap=max(1, int(n_bootstrap)), ci=float(ci), seed=int(seed))
    return ClusteringStability(
        mean_ari=float(mean_ari),
        ari_ci_low=float(lo),
        ari_ci_high=float(hi),
        n_bootstrap=int(len(aris)),
    )


def cluster_summary_rows(
    matrix: FeatureEffectMatrix,
    clusters: Mapping[int, Sequence[int]],
    *,
    stability: ClusteringStability,
) -> List[Dict[str, float | int]]:
    out: List[Dict[str, float | int]] = []
    vectors = {int(fid): _vector(matrix, int(fid)) for fid in matrix.feature_ids}
    for cid, fids in sorted(clusters.items(), key=lambda kv: kv[0]):
        f = [int(x) for x in fids]
        if not f:
            continue
        mat = np.stack([vectors[x] for x in f], axis=0)
        centroid = mat.mean(axis=0)
        sims = [_cosine(vectors[x], centroid) for x in f]
        out.append(
            {
                "cluster_id": int(cid),
                "size": int(len(f)),
                "mean_centroid_similarity": float(np.mean(sims)),
                "stability_mean_ari": float(stability.mean_ari),
                "stability_ari_ci_low": float(stability.ari_ci_low),
                "stability_ari_ci_high": float(stability.ari_ci_high),
                "stability_n_bootstrap": int(stability.n_bootstrap),
            }
        )
    return out
