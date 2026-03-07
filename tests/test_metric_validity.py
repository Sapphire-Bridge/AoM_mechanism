from __future__ import annotations

import math

import pytest

from aom.metrics.composite import compute_composite_metric
from aom.stats import bootstrap_ci, bootstrap_ci_metric


def test_bootstrap_ci_metric_empty_is_invalid_nan():
    mv, lo, hi = bootstrap_ci_metric([], n_bootstrap=10, ci=0.95, seed=0)
    assert mv.n == 0
    assert mv.valid is False
    assert mv.reason == "empty sample"
    assert math.isnan(mv.value)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_bootstrap_ci_metric_validates_args():
    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap_ci_metric([1.0], n_bootstrap=0, ci=0.95, seed=0)
    with pytest.raises(ValueError, match="ci must be in"):
        bootstrap_ci_metric([1.0], n_bootstrap=10, ci=1.0, seed=0)


def test_bootstrap_ci_empty_returns_nan():
    mu, lo, hi = bootstrap_ci([], n_bootstrap=10, ci=0.95, seed=0)
    assert math.isnan(mu)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_compute_composite_metric_missing_default_nan():
    disamb = {"accuracy": 0.5}
    cf = {"shift_direction_accuracy": float("nan")}
    coh = {"constraint_accuracy": 0.75}
    mv = compute_composite_metric(disamb, cf, coh)
    assert mv.valid is False
    assert math.isnan(mv.value)
    assert mv.reason is not None and "cf.shift_direction_accuracy" in mv.reason


def test_compute_composite_metric_ignore_missing():
    disamb = {"accuracy": 0.5}
    cf = {"shift_direction_accuracy": float("nan")}
    coh = {"constraint_accuracy": 0.75}
    mv = compute_composite_metric(disamb, cf, coh, missing_policy="ignore")
    assert mv.valid is True
    assert mv.value == pytest.approx((0.5 + 0.75) / 2.0)
    assert mv.reason is not None and "cf.shift_direction_accuracy" in mv.reason


def test_compute_composite_metric_fail_missing():
    disamb = {"accuracy": 0.5}
    cf = {"shift_direction_accuracy": float("nan")}
    coh = {"constraint_accuracy": 0.75}
    with pytest.raises(ValueError, match="missing/invalid primary metrics"):
        compute_composite_metric(disamb, cf, coh, missing_policy="fail")
