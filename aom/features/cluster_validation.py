from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np

from aom.features.effect_matrix import FeatureEffectMatrix
from aom.stats import bootstrap_ci


def _cluster_effect(matrix: FeatureEffectMatrix, feature_ids: Sequence[int]) -> float:
    vals: List[float] = []
    for fid in feature_ids:
        row = matrix.effects.get(int(fid), {})
        for cid in matrix.condition_ids:
            v = float(row.get(str(cid), float("nan")))
            if np.isfinite(v):
                vals.append(abs(float(v)))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


@dataclass(frozen=True)
class ClusterValidationRow:
    cluster_id: int
    size: int
    effect: float
    null_mean: float
    null_ci_low: float
    null_ci_high: float
    p_value: float

    def to_row(self) -> Dict[str, float | int]:
        return {
            "cluster_id": int(self.cluster_id),
            "size": int(self.size),
            "effect": float(self.effect),
            "null_mean": float(self.null_mean),
            "null_ci_low": float(self.null_ci_low),
            "null_ci_high": float(self.null_ci_high),
            "p_value": float(self.p_value),
        }


def validate_clusters_against_null(
    matrix: FeatureEffectMatrix,
    clusters: Mapping[int, Sequence[int]],
    *,
    n_null: int = 500,
    seed: int = 42,
    ci: float = 0.95,
) -> List[ClusterValidationRow]:
    feature_universe = [int(fid) for fid in matrix.feature_ids]
    rng = np.random.RandomState(int(seed))
    out: List[ClusterValidationRow] = []

    for cid, fids in sorted(clusters.items(), key=lambda kv: kv[0]):
        members = [int(x) for x in fids]
        k = int(len(members))
        if k <= 0:
            continue
        effect = _cluster_effect(matrix, members)
        null_vals: List[float] = []
        for _ in range(int(n_null)):
            if k >= len(feature_universe):
                sample = list(feature_universe)
            else:
                idx = rng.choice(len(feature_universe), size=k, replace=False)
                sample = [feature_universe[int(i)] for i in idx]
            val = _cluster_effect(matrix, sample)
            if np.isfinite(val):
                null_vals.append(float(val))
        null_mean, null_lo, null_hi = bootstrap_ci(
            null_vals,
            n_bootstrap=max(1, min(500, int(n_null))),
            ci=float(ci),
            seed=int(seed),
        )
        if null_vals:
            p_value = float(sum(1 for x in null_vals if x >= float(effect)) / len(null_vals))
        else:
            p_value = float("nan")
        out.append(
            ClusterValidationRow(
                cluster_id=int(cid),
                size=int(k),
                effect=float(effect),
                null_mean=float(null_mean),
                null_ci_low=float(null_lo),
                null_ci_high=float(null_hi),
                p_value=float(p_value),
            )
        )
    return out
