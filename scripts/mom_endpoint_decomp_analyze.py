from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.metrics.primary_logodds import PRIMARY_LOGODDS_RESIDUAL_TOL_DEFAULT


PRIMARY_FIELDS = (
    "P_E_base",
    "P_O_base",
    "P_rest_base",
    "P_E_patch_A",
    "P_O_patch_A",
    "P_rest_patch_A",
    "P_E_patch_C",
    "P_O_patch_C",
    "P_rest_patch_C",
    "dlogPE_A",
    "dlogPO_A",
    "dlogPE_C",
    "dlogPO_C",
    "dPE_A",
    "dPO_A",
    "dPrest_A",
    "dPE_C",
    "dPO_C",
    "dPrest_C",
    "d_CA_logPE",
    "d_CA_logPO",
    "primary_delta_m_from_logodds_A",
    "primary_delta_m_from_logodds_C",
    "primary_delta_m_residual_A",
    "primary_delta_m_residual_C",
)

REQUIRED_PRIMARY_METADATA_FIELDS = (
    "primary_static_applicable",
    "primary_logodds_applicable",
    "primary_cancellation_pass",
    "primary_labels_binary",
    "primary_expected_all_single",
    "primary_other_all_single",
    "primary_token_sets_disjoint",
    "primary_candidate_count_equal",
    "primary_delta_m_residual_A",
    "primary_delta_m_residual_C",
)


def _b(s: object) -> bool:
    v = str(s).strip().lower()
    return v in {"1", "true", "yes"}


def _f(s: object) -> float:
    try:
        return float(s)
    except Exception:
        return float("nan")


def _finite_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _load_optional_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _tokenizer_file_hashes(tokenizer_root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
    ):
        fp = tokenizer_root / name
        if fp.exists() and fp.is_file():
            out[name] = _sha256_file(fp)
    return out


def _percentile_interval(values: Sequence[float], ci: float) -> Tuple[float, float]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan"), float("nan")
    alpha = max(0.0, min(1.0, float(ci)))
    lo_i = int(((1.0 - alpha) / 2.0) * len(vals))
    hi_i = max(lo_i, int(((1.0 + alpha) / 2.0) * len(vals)) - 1)
    lo_i = max(0, min(lo_i, len(vals) - 1))
    hi_i = max(0, min(hi_i, len(vals) - 1))
    return float(vals[lo_i]), float(vals[hi_i])


def _metric_bootstrap(
    values: Sequence[float],
    *,
    bootstrap_n: int,
    ci: float,
    seed: int,
) -> Dict[str, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_gt_0": float("nan")}
    rng = random.Random(int(seed))
    boot: List[float] = []
    for _ in range(int(bootstrap_n)):
        sampled = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        boot.append(float(sum(sampled) / len(sampled)))
    lo, hi = _percentile_interval(boot, ci=ci)
    return {
        "n": int(len(vals)),
        "mean": float(sum(vals) / len(vals)),
        "ci_low": lo,
        "ci_high": hi,
        "p_gt_0": float(sum(1 for x in boot if x > 0.0) / len(boot)),
    }


@dataclass(frozen=True)
class PairAggregate:
    layer: int
    pair_id: str
    n_rows: int
    n_primary_rows: int
    primary_static_pair: bool
    effect_a: float
    effect_c: float
    d_ca: float
    crr: float
    d_ca_diag_logz: float
    d_ca_logodds_residual: float
    metrics: Dict[str, float]


def _load_rows(path: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = tuple(str(x) for x in (reader.fieldnames or ()))
        missing = [
            field
            for field in (
                "analysis_included",
                "pair_id",
                "layer",
                "effect_A",
                "effect_C",
                "decomp_exp_delta_logz_mean_A",
                "decomp_other_delta_logz_mean_A",
                "decomp_exp_delta_logz_mean_C",
                "decomp_other_delta_logz_mean_C",
                *PRIMARY_FIELDS,
                *REQUIRED_PRIMARY_METADATA_FIELDS,
            )
            if field not in fieldnames
        ]
        if missing:
            raise ValueError(
                "Missing required columns for endpoint-native analysis. "
                f"Expected new-schema comparability CSV; missing={missing}"
            )
        for row in reader:
            if _b(row.get("analysis_included", "False")):
                out.append(dict(row))
    if not out:
        raise ValueError(f"No analysis_included rows found in {path}")
    return out


def _build_pair_aggregates(rows: Sequence[Mapping[str, str]], *, ratio_den_eps: float) -> List[PairAggregate]:
    grouped: Dict[Tuple[int, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), str(row["pair_id"]))].append(row)

    pair_aggs: List[PairAggregate] = []
    for (layer, pair_id), grows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        effect_a = _finite_mean(_f(r.get("effect_A", "nan")) for r in grows)
        effect_c = _finite_mean(_f(r.get("effect_C", "nan")) for r in grows)
        d_ca = float(effect_c - effect_a) if math.isfinite(effect_a) and math.isfinite(effect_c) else float("nan")
        crr = float(effect_c / effect_a) if math.isfinite(effect_a) and abs(effect_a) > float(ratio_den_eps) else float("nan")

        primary_static_flags = [_b(r.get("primary_static_applicable", False)) for r in grows]
        primary_static_pair = bool(primary_static_flags and all(primary_static_flags))

        primary_rows = [r for r in grows if _b(r.get("primary_logodds_applicable", False))]
        primary_pair_complete = bool(len(primary_rows) == len(grows) and len(grows) > 0)

        metrics: Dict[str, float] = {}
        for field in PRIMARY_FIELDS:
            metrics[field] = (
                _finite_mean(_f(r.get(field, "nan")) for r in primary_rows) if primary_pair_complete else float("nan")
            )

        if primary_pair_complete:
            d_ca_diag_logz = _finite_mean(
                (
                    -(_f(r.get("decomp_exp_delta_logz_mean_C", "nan")) - _f(r.get("decomp_other_delta_logz_mean_C", "nan")))
                    - (
                        -(
                            _f(r.get("decomp_exp_delta_logz_mean_A", "nan"))
                            - _f(r.get("decomp_other_delta_logz_mean_A", "nan"))
                        )
                    )
                )
                for r in primary_rows
            )
            d_ca_from_logodds = float(metrics["d_CA_logPE"] - metrics["d_CA_logPO"]) if (
                math.isfinite(metrics["d_CA_logPE"]) and math.isfinite(metrics["d_CA_logPO"])
            ) else float("nan")
            d_ca_logodds_residual = float(d_ca - d_ca_from_logodds) if (
                math.isfinite(d_ca) and math.isfinite(d_ca_from_logodds)
            ) else float("nan")
        else:
            d_ca_diag_logz = float("nan")
            d_ca_logodds_residual = float("nan")

        pair_aggs.append(
            PairAggregate(
                layer=int(layer),
                pair_id=str(pair_id),
                n_rows=int(len(grows)),
                n_primary_rows=int(len(primary_rows)),
                primary_static_pair=bool(primary_static_pair),
                effect_a=float(effect_a),
                effect_c=float(effect_c),
                d_ca=float(d_ca),
                crr=float(crr),
                d_ca_diag_logz=float(d_ca_diag_logz),
                d_ca_logodds_residual=float(d_ca_logodds_residual),
                metrics=metrics,
            )
        )

    if not pair_aggs:
        raise ValueError("No pair aggregates built from analysis rows.")
    return pair_aggs


def _bootstrap_effects(
    pairs: Sequence[PairAggregate],
    *,
    bootstrap_n: int,
    ci: float,
    seed: int,
    ratio_den_eps: float,
) -> Dict[str, object]:
    if not pairs:
        nan = float("nan")
        return {
            "n_pairs": 0,
            "effect_a": {"mean": nan, "ci_low": nan, "ci_high": nan},
            "effect_c": {"mean": nan, "ci_low": nan, "ci_high": nan},
            "ddm": {"mean": nan, "ci_low": nan, "ci_high": nan},
            "crr": {"mean": nan, "ci_low": nan, "ci_high": nan},
            "p_ddm_gt_0": nan,
            "p_crr_gt_1": nan,
        }

    point_a = _finite_mean(p.effect_a for p in pairs)
    point_c = _finite_mean(p.effect_c for p in pairs)
    point_ddm = float(point_c - point_a)
    point_crr = float("nan") if abs(point_a) <= float(ratio_den_eps) else float(point_c / point_a)

    rng = random.Random(int(seed))
    boot_a: List[float] = []
    boot_c: List[float] = []
    boot_ddm: List[float] = []
    boot_crr: List[float] = []
    for _ in range(int(bootstrap_n)):
        sampled = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        m_a = _finite_mean(p.effect_a for p in sampled)
        m_c = _finite_mean(p.effect_c for p in sampled)
        m_ddm = float(m_c - m_a)
        boot_a.append(m_a)
        boot_c.append(m_c)
        boot_ddm.append(m_ddm)
        if abs(m_a) > float(ratio_den_eps):
            boot_crr.append(float(m_c / m_a))

    a_lo, a_hi = _percentile_interval(boot_a, ci=ci)
    c_lo, c_hi = _percentile_interval(boot_c, ci=ci)
    ddm_lo, ddm_hi = _percentile_interval(boot_ddm, ci=ci)
    crr_lo, crr_hi = _percentile_interval(boot_crr, ci=ci) if boot_crr else (float("nan"), float("nan"))

    return {
        "n_pairs": int(len(pairs)),
        "effect_a": {"mean": point_a, "ci_low": a_lo, "ci_high": a_hi},
        "effect_c": {"mean": point_c, "ci_low": c_lo, "ci_high": c_hi},
        "ddm": {"mean": point_ddm, "ci_low": ddm_lo, "ci_high": ddm_hi},
        "crr": {"mean": point_crr, "ci_low": crr_lo, "ci_high": crr_hi},
        "p_ddm_gt_0": float(sum(1 for x in boot_ddm if x > 0.0) / len(boot_ddm)),
        "p_crr_gt_1": float(sum(1 for x in boot_crr if x > 1.0) / len(boot_crr)) if boot_crr else float("nan"),
    }


def _split_pairs(pairs: Sequence[PairAggregate], split: str) -> List[PairAggregate]:
    if split == "all_pairs":
        return list(pairs)
    if split == "primary_applicable_pairs":
        return [p for p in pairs if bool(p.primary_static_pair)]
    if split == "non_primary_applicable_pairs":
        return [p for p in pairs if not bool(p.primary_static_pair)]
    raise ValueError(f"Unsupported split {split!r}")


def _layer_summary(
    pairs: Sequence[PairAggregate],
    *,
    bootstrap_n: int,
    ci: float,
    seed: int,
    ratio_den_eps: float,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for split in ("all_pairs", "primary_applicable_pairs", "non_primary_applicable_pairs"):
        spairs = _split_pairs(pairs, split)
        s: Dict[str, object] = _bootstrap_effects(
            spairs,
            bootstrap_n=bootstrap_n,
            ci=ci,
            seed=seed,
            ratio_den_eps=ratio_den_eps,
        )
        s["d_ca_logpe"] = _metric_bootstrap(
            [p.metrics["d_CA_logPE"] for p in spairs], bootstrap_n=bootstrap_n, ci=ci, seed=seed + 11
        )
        s["d_ca_logpo"] = _metric_bootstrap(
            [p.metrics["d_CA_logPO"] for p in spairs], bootstrap_n=bootstrap_n, ci=ci, seed=seed + 13
        )
        s["d_ca_diag_logz"] = _metric_bootstrap(
            [p.d_ca_diag_logz for p in spairs], bootstrap_n=bootstrap_n, ci=ci, seed=seed + 17
        )
        s["d_ca_logodds_residual"] = _metric_bootstrap(
            [p.d_ca_logodds_residual for p in spairs], bootstrap_n=bootstrap_n, ci=ci, seed=seed + 19
        )
        abs_residuals = [abs(float(p.d_ca_logodds_residual)) for p in spairs if math.isfinite(float(p.d_ca_logodds_residual))]
        s["d_ca_logodds_residual_abs_max"] = float(max(abs_residuals)) if abs_residuals else float("nan")
        s["dpe_a"] = _metric_bootstrap([p.metrics["dPE_A"] for p in spairs], bootstrap_n=bootstrap_n, ci=ci, seed=seed + 23)
        s["dpo_a"] = _metric_bootstrap([p.metrics["dPO_A"] for p in spairs], bootstrap_n=bootstrap_n, ci=ci, seed=seed + 29)
        s["dprest_a"] = _metric_bootstrap(
            [p.metrics["dPrest_A"] for p in spairs], bootstrap_n=bootstrap_n, ci=ci, seed=seed + 31
        )
        s["dpe_c"] = _metric_bootstrap([p.metrics["dPE_C"] for p in spairs], bootstrap_n=bootstrap_n, ci=ci, seed=seed + 37)
        s["dpo_c"] = _metric_bootstrap([p.metrics["dPO_C"] for p in spairs], bootstrap_n=bootstrap_n, ci=ci, seed=seed + 41)
        s["dprest_c"] = _metric_bootstrap(
            [p.metrics["dPrest_C"] for p in spairs], bootstrap_n=bootstrap_n, ci=ci, seed=seed + 43
        )
        out[split] = s
    return out


def _claim_routing(
    *,
    layer_result: Mapping[str, object],
    residual_tol: float,
    diag_logz_min_abs_mean: float,
) -> Dict[str, object]:
    primary = layer_result.get("primary_applicable_pairs", {})
    ddm = primary.get("ddm", {})
    d_ca_diag_logz = primary.get("d_ca_diag_logz", {})
    d_ca_res = primary.get("d_ca_logodds_residual", {})

    ddm_mean = float(ddm.get("mean", float("nan")))
    ddm_p_gt_0 = float(primary.get("p_ddm_gt_0", float("nan")))
    diag_logz_mean = float(d_ca_diag_logz.get("mean", float("nan")))
    diag_logz_lo = float(d_ca_diag_logz.get("ci_low", float("nan")))
    diag_logz_hi = float(d_ca_diag_logz.get("ci_high", float("nan")))
    residual_mean = float(d_ca_res.get("mean", float("nan")))
    residual_abs_max = float(primary.get("d_ca_logodds_residual_abs_max", float("nan")))
    residual_abs_ok = bool(math.isfinite(residual_mean) and abs(residual_mean) <= float(residual_tol))
    residual_max_ok = bool(math.isfinite(residual_abs_max) and residual_abs_max <= float(residual_tol))
    diag_logz_excludes_zero = bool(
        math.isfinite(diag_logz_lo) and math.isfinite(diag_logz_hi) and (diag_logz_lo > 0.0 or diag_logz_hi < 0.0)
    )
    diag_logz_material = bool(math.isfinite(diag_logz_mean) and abs(diag_logz_mean) >= float(diag_logz_min_abs_mean))

    return {
        "supports_primary_diag_logz_nonzero": bool(diag_logz_excludes_zero and diag_logz_material),
        "supports_candidate_set_logodds_accounting": bool(residual_abs_ok and residual_max_ok),
        "primary_ddm_mean": ddm_mean,
        "primary_p_ddm_gt_0": ddm_p_gt_0,
        "primary_d_ca_diag_logz_mean": diag_logz_mean,
        "primary_d_ca_diag_logz_ci_low": diag_logz_lo,
        "primary_d_ca_diag_logz_ci_high": diag_logz_hi,
        "primary_d_ca_logodds_residual_mean": residual_mean,
        "primary_d_ca_logodds_residual_abs_max": residual_abs_max,
    }


def _write_pair_csv(path: Path, pairs: Sequence[PairAggregate]) -> None:
    fields = [
        "layer",
        "pair_id",
        "n_rows",
        "n_primary_rows",
        "primary_static_pair",
        "effect_A",
        "effect_C",
        "d_CA",
        "crr",
        "d_CA_diag_logz",
        "d_CA_logodds_residual",
    ] + list(PRIMARY_FIELDS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in pairs:
            row = {
                "layer": int(p.layer),
                "pair_id": str(p.pair_id),
                "n_rows": int(p.n_rows),
                "n_primary_rows": int(p.n_primary_rows),
                "primary_static_pair": bool(p.primary_static_pair),
                "effect_A": float(p.effect_a),
                "effect_C": float(p.effect_c),
                "d_CA": float(p.d_ca),
                "crr": float(p.crr),
                "d_CA_diag_logz": float(p.d_ca_diag_logz),
                "d_CA_logodds_residual": float(p.d_ca_logodds_residual),
            }
            for field in PRIMARY_FIELDS:
                row[field] = float(p.metrics.get(field, float("nan")))
            writer.writerow(row)


def _fmt(v: float, digits: int = 3) -> str:
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f}"


def _render_md(summary: Mapping[str, object]) -> str:
    lines: List[str] = []
    lines.append("# MoM Endpoint Decomposition: R1 Summary")
    lines.append("")
    lines.append("Primary inferential object: paired `ΔΔm = Δm_C - Δm_A`.")
    lines.append("")
    source = summary.get("source", {})
    hashes = summary.get("hashes", {})
    lines.append("## Provenance")
    lines.append(f"- comparability_csv: `{source.get('comparability_csv', '')}`")
    if str(source.get("comparability_summary", "")).strip():
        lines.append(f"- comparability_summary: `{source.get('comparability_summary', '')}`")
    if str(source.get("disamb_path", "")).strip():
        lines.append(f"- disamb_path: `{source.get('disamb_path', '')}`")
    if str(source.get("tokenizer_name_or_path", "")).strip():
        lines.append(f"- tokenizer_name_or_path: `{source.get('tokenizer_name_or_path', '')}`")
        lines.append(f"- tokenizer_name_or_path_resolved: `{source.get('tokenizer_name_or_path_resolved', '')}`")
    lines.append(f"- comparability_csv_sha256: `{hashes.get('comparability_csv_sha256', '')}`")
    if str(hashes.get("comparability_summary_sha256", "")).strip():
        lines.append(f"- comparability_summary_sha256: `{hashes.get('comparability_summary_sha256', '')}`")
    if str(hashes.get("disamb_sha256", "")).strip():
        lines.append(f"- disamb_sha256: `{hashes.get('disamb_sha256', '')}`")
    lines.append(f"- analysis_script_sha256: `{hashes.get('analysis_script_sha256', '')}`")
    lines.append(f"- git_commit: `{hashes.get('git_commit', '')}`")
    tok_hashes = dict(hashes.get("tokenizer_file_hashes", {}))
    lines.append(f"- tokenizer_file_hashes: `{json.dumps(tok_hashes, sort_keys=True) if tok_hashes else 'none'}`")
    lines.append("")
    lines.append("## Claim Routing")
    routing = summary["claim_routing_by_layer"]
    for layer in sorted(routing.keys(), key=lambda x: int(x)):
        r = routing[layer]
        lines.append(f"- L{layer}:")
        lines.append(f"  - supports_primary_diag_logz_nonzero: `{r['supports_primary_diag_logz_nonzero']}`")
        lines.append(f"  - supports_candidate_set_logodds_accounting: `{r['supports_candidate_set_logodds_accounting']}`")
        lines.append(
            f"  - primary-applicable ΔΔm mean / P(>0): `{_fmt(float(r['primary_ddm_mean']))}` / `{_fmt(float(r['primary_p_ddm_gt_0']))}`"
        )
        lines.append(
            f"  - primary-applicable d_CA_diag_logz mean [CI]: `{_fmt(float(r['primary_d_ca_diag_logz_mean']), 6)}` "
            f"[`{_fmt(float(r['primary_d_ca_diag_logz_ci_low']), 6)}`, `{_fmt(float(r['primary_d_ca_diag_logz_ci_high']), 6)}`]"
        )
    lines.append("")
    lines.append("## Layered Results (Primary-applicable split)")
    lines.append(
        "| layer | n_pairs | ΔΔm mean [CI] | P(ΔΔm>0) | d_CA_logPE mean | d_CA_logPO mean | d_CA_logodds_residual mean |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for layer in sorted(summary["results_by_layer"].keys(), key=lambda x: int(x)):
        primary = summary["results_by_layer"][layer]["primary_applicable_pairs"]
        ddm = primary["ddm"]
        d_ca_logpe = primary["d_ca_logpe"]
        d_ca_logpo = primary["d_ca_logpo"]
        d_ca_res = primary["d_ca_logodds_residual"]
        lines.append(
            "| {layer} | {n} | {m} [{lo}, {hi}] | {p} | {pe} | {po} | {res} |".format(
                layer=layer,
                n=int(primary["n_pairs"]),
                m=_fmt(float(ddm["mean"])),
                lo=_fmt(float(ddm["ci_low"])),
                hi=_fmt(float(ddm["ci_high"])),
                p=_fmt(float(primary["p_ddm_gt_0"])),
                pe=_fmt(float(d_ca_logpe["mean"])),
                po=_fmt(float(d_ca_logpo["mean"])),
                res=_fmt(float(d_ca_res["mean"]), 6),
            )
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    in_default = root / "results" / "mom_endpoint_decomp_r1" / "comparability_endpoint_v2.csv"
    out_dir = root / "results" / "mom_endpoint_decomp_r1"
    p = argparse.ArgumentParser(description="Analyze endpoint-native candidate-set terms for MoM R1.")
    p.add_argument("--comparability_csv", type=Path, default=in_default)
    p.add_argument("--comparability_summary", type=Path, default=None)
    p.add_argument("--disamb_path", type=Path, default=None)
    p.add_argument("--tokenizer_name_or_path", type=str, default="")
    p.add_argument("--bootstrap_n", type=int, default=5000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratio_den_eps", type=float, default=1e-8)
    p.add_argument("--primary_residual_tol", type=float, default=float(PRIMARY_LOGODDS_RESIDUAL_TOL_DEFAULT))
    p.add_argument("--diag_logz_min_abs_mean", type=float, default=1e-3)
    p.add_argument("--out_pair_csv", type=Path, default=out_dir / "endpoint_pair_aggregates_v2.csv")
    p.add_argument("--out_json", type=Path, default=out_dir / "endpoint_decomp_summary_v2.json")
    p.add_argument("--out_md", type=Path, default=out_dir / "endpoint_decomp_summary_v2.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.comparability_csv)
    pair_aggs = _build_pair_aggregates(rows, ratio_den_eps=float(args.ratio_den_eps))

    summary_path = args.comparability_summary
    if summary_path is None:
        sidecar = args.comparability_csv.with_suffix(".summary.json")
        if sidecar.exists() and sidecar.is_file():
            summary_path = sidecar
    summary_obj = _load_optional_json(summary_path)

    disamb_path = args.disamb_path
    if disamb_path is None and isinstance(summary_obj, dict):
        cand = summary_obj.get("disamb_path")
        if isinstance(cand, str) and cand.strip():
            disamb_path = Path(cand)

    tokenizer_name_or_path = str(args.tokenizer_name_or_path).strip()
    if not tokenizer_name_or_path and isinstance(summary_obj, dict):
        cand = summary_obj.get("model_name_or_path")
        if isinstance(cand, str) and cand.strip():
            tokenizer_name_or_path = cand.strip()
    tokenizer_path = Path(tokenizer_name_or_path) if tokenizer_name_or_path else None
    tokenizer_file_hashes: Dict[str, str] = {}
    tokenizer_name_or_path_resolved = tokenizer_name_or_path
    if tokenizer_path is not None and tokenizer_path.exists() and tokenizer_path.is_dir():
        tokenizer_name_or_path_resolved = str(tokenizer_path.resolve())
        tokenizer_file_hashes = _tokenizer_file_hashes(tokenizer_path)

    layers = sorted({int(p.layer) for p in pair_aggs})
    results_by_layer: Dict[str, Dict[str, object]] = {}
    claim_routing_by_layer: Dict[str, Dict[str, object]] = {}
    for layer in layers:
        lpairs = [p for p in pair_aggs if int(p.layer) == int(layer)]
        layer_result = _layer_summary(
            lpairs,
            bootstrap_n=int(args.bootstrap_n),
            ci=float(args.ci),
            seed=int(args.seed) + int(layer) * 100,
            ratio_den_eps=float(args.ratio_den_eps),
        )
        results_by_layer[str(int(layer))] = layer_result
        claim_routing_by_layer[str(int(layer))] = _claim_routing(
            layer_result=layer_result,
            residual_tol=float(args.primary_residual_tol),
            diag_logz_min_abs_mean=float(args.diag_logz_min_abs_mean),
        )

    primary_rows = [r for r in rows if _b(r.get("primary_logodds_applicable", False))]
    primary_cancel_fail = sum(1 for r in primary_rows if not _b(r.get("primary_cancellation_pass", False)))
    primary_residual_violations = 0
    for r in primary_rows:
        ra = abs(_f(r.get("primary_delta_m_residual_A", "nan")))
        rc = abs(_f(r.get("primary_delta_m_residual_C", "nan")))
        if (math.isfinite(ra) and ra > float(args.primary_residual_tol)) or (
            math.isfinite(rc) and rc > float(args.primary_residual_tol)
        ):
            primary_residual_violations += 1

    summary: Dict[str, object] = {
        "source": {
            "comparability_csv": str(args.comparability_csv),
            "comparability_summary": str(summary_path) if summary_path is not None else "",
            "disamb_path": str(disamb_path) if disamb_path is not None else "",
            "tokenizer_name_or_path": tokenizer_name_or_path,
            "tokenizer_name_or_path_resolved": tokenizer_name_or_path_resolved,
        },
        "hashes": {
            "comparability_csv_sha256": _sha256_file(args.comparability_csv),
            "comparability_summary_sha256": (
                _sha256_file(summary_path) if summary_path is not None and summary_path.exists() else ""
            ),
            "disamb_sha256": _sha256_file(disamb_path) if disamb_path is not None and disamb_path.exists() else "",
            "tokenizer_file_hashes": tokenizer_file_hashes,
            "analysis_script_sha256": _sha256_file(Path(__file__).resolve()),
            "git_commit": _git_commit(ROOT),
        },
        "config": {
            "bootstrap_n": int(args.bootstrap_n),
            "ci": float(args.ci),
            "seed": int(args.seed),
            "ratio_den_eps": float(args.ratio_den_eps),
            "primary_residual_tol": float(args.primary_residual_tol),
            "diag_logz_min_abs_mean": float(args.diag_logz_min_abs_mean),
        },
        "counts": {
            "n_rows_analysis_included": int(len(rows)),
            "n_rows_primary_logodds_applicable": int(len(primary_rows)),
            "n_rows_primary_cancellation_fail": int(primary_cancel_fail),
            "n_rows_primary_residual_violations": int(primary_residual_violations),
            "n_pairs_total": int(len(pair_aggs)),
        },
        "results_by_layer": results_by_layer,
        "claim_routing_by_layer": claim_routing_by_layer,
    }

    _write_pair_csv(args.out_pair_csv, pair_aggs)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_md(summary), encoding="utf-8")
    print(str(args.out_pair_csv))
    print(str(args.out_json))
    print(str(args.out_md))


if __name__ == "__main__":
    main()
