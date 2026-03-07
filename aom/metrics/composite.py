from __future__ import annotations

import math
from typing import Any, Mapping

from aom.stats import MetricValue


def compute_composite_metric(
    disamb: Mapping[str, Any],
    cf: Mapping[str, Any],
    coh: Mapping[str, Any],
    *,
    missing_policy: str = "nan",
) -> MetricValue:
    """
    Simple composite for plotting: average of primary accuracies.

    Primary metrics:
    - AoM-DISAMB: disamb["accuracy"]
    - AoM-CF: cf["shift_direction_accuracy"]
    - AoM-COH: coh["constraint_accuracy"]
    """
    if missing_policy not in {"fail", "nan", "ignore"}:
        raise ValueError("missing_policy must be one of: fail, nan, ignore")

    specs = [
        ("disamb", disamb, "accuracy"),
        ("cf", cf, "shift_direction_accuracy"),
        ("coh", coh, "constraint_accuracy"),
    ]
    vals: list[float] = []
    missing: list[str] = []
    for scope, mapping, key in specs:
        v = mapping.get(key, None)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            missing.append(f"{scope}.{key}")
            continue
        fv = float(v)
        if not math.isfinite(fv):
            missing.append(f"{scope}.{key}")
            continue
        vals.append(fv)

    if missing:
        reason = "missing/invalid primary metrics: " + ", ".join(missing)
        if missing_policy == "ignore" and vals:
            return MetricValue(value=float(sum(vals) / len(vals)), n=len(vals), valid=True, reason=reason)
        if missing_policy == "fail":
            raise ValueError(reason)
        nan = float("nan")
        return MetricValue(value=nan, n=len(vals), valid=False, reason=reason)

    if not vals:
        if missing_policy == "fail":
            raise ValueError("no primary metrics available for composite")
        nan = float("nan")
        return MetricValue(value=nan, n=0, valid=False, reason="no primary metrics available for composite")

    return MetricValue(value=float(sum(vals) / len(vals)), n=len(vals), valid=True, reason=None)


def compute_composite(
    disamb: Mapping[str, Any],
    cf: Mapping[str, Any],
    coh: Mapping[str, Any],
    *,
    missing_policy: str = "nan",
) -> float:
    return float(compute_composite_metric(disamb, cf, coh, missing_policy=missing_policy).value)
