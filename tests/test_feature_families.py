from __future__ import annotations

from aom.features.cluster_validation import validate_clusters_against_null
from aom.features.clustering import bootstrap_stability, cluster_feature_families
from aom.features.effect_matrix import FeatureEffectMatrix


def _synthetic_matrix() -> FeatureEffectMatrix:
    # Two clear families: {0,1} and {2,3}.
    effects = {
        0: {"c0": 1.0, "c1": 1.0, "c2": 0.0, "c3": 0.0},
        1: {"c0": 0.9, "c1": 1.1, "c2": 0.0, "c3": 0.1},
        2: {"c0": 0.0, "c1": 0.0, "c2": 1.0, "c3": 1.0},
        3: {"c0": 0.1, "c1": 0.0, "c2": 0.9, "c3": 1.1},
    }
    return FeatureEffectMatrix(
        feature_ids=(0, 1, 2, 3),
        condition_ids=("c0", "c1", "c2", "c3"),
        effects=effects,
    )


def test_cluster_recovery_and_stability():
    matrix = _synthetic_matrix()
    clusters = cluster_feature_families(matrix, similarity_threshold=0.8)
    sizes = sorted(len(v) for v in clusters.values())
    assert sizes == [2, 2]

    stability = bootstrap_stability(matrix, similarity_threshold=0.8, n_bootstrap=100, seed=0, ci=0.95)
    assert stability.n_bootstrap > 0
    assert stability.mean_ari >= 0.8


def test_cluster_validation_can_fail():
    matrix = FeatureEffectMatrix(
        feature_ids=(0, 1, 2, 3),
        condition_ids=("c0", "c1", "c2", "c3"),
        effects={
            0: {"c0": 1.0, "c1": 1.0, "c2": 1.0, "c3": 1.0},
            1: {"c0": 0.9, "c1": 1.1, "c2": 1.0, "c3": 1.0},
            2: {"c0": 0.0, "c1": 0.0, "c2": 0.0, "c3": 0.0},
            3: {"c0": 0.05, "c1": 0.0, "c2": 0.0, "c3": 0.0},
        },
    )
    clusters = {0: [0, 1], 1: [2, 3]}
    rows = validate_clusters_against_null(matrix, clusters, n_null=200, seed=0, ci=0.95)
    assert rows
    # At least one cluster should not beat null strongly.
    assert any(float(r.p_value) >= 0.05 for r in rows)
