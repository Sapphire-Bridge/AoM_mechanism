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
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.metrics.primary_logodds import choices_token_ids_from_strings, evaluate_primary_applicability


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _fmt(v: float, digits: int = 3) -> str:
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f}"


@dataclass(frozen=True)
class PairEffect:
    pair_id: str
    effect_a: float
    effect_c: float

    @property
    def ddm(self) -> float:
        return float(self.effect_c - self.effect_a)


def _load_analysis_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("analysis_included", "False")) != "True":
                continue
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"No analysis_included rows in {path}")
    return rows


def _infer_tokenizer_name(summary_json: Path, explicit_name: str) -> str:
    if explicit_name.strip():
        return explicit_name.strip()
    with summary_json.open(encoding="utf-8") as f:
        summary = json.load(f)
    model_name = str(summary.get("model_name_or_path", "")).strip()
    if not model_name:
        raise ValueError(
            f"Cannot infer tokenizer path from summary: {summary_json}. "
            "Pass --tokenizer_name_or_path explicitly."
        )
    return model_name


def _pair_primary_profile(disamb_path: Path, tokenizer_name_or_path: str) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, local_files_only=True)
    out: Dict[str, Dict[str, object]] = {}
    with disamb_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            pair_id = str(it["pair_id"])
            choices = {str(k): list(v) for k, v in dict(it["choices"]).items()}
            choices_token_ids = choices_token_ids_from_strings(tok, choices)

            token_lens: List[int] = [len(seq) for seqs in choices_token_ids.values() for seq in seqs]
            if not token_lens:
                raise ValueError(f"Pair {pair_id} has no continuation tokens")
            n_multi = int(sum(1 for x in token_lens if x > 1))

            def _direction_primary(expected_label: str) -> Tuple[bool, str]:
                if str(expected_label) not in choices_token_ids:
                    return False, "expected_label_missing"
                other_labels = [lab for lab in choices_token_ids.keys() if str(lab) != str(expected_label)]
                other_label = str(other_labels[0]) if other_labels else ""
                # Phase-1 baseline only checks static applicability on current DISAMB; these are
                # synthetic equality markers, not reconstructed prompt-boundary positions.
                scored_positions = [0] * (
                    len(choices_token_ids.get(str(expected_label), ())) + len(choices_token_ids.get(str(other_label), ()))
                )
                app = evaluate_primary_applicability(
                    choices_token_ids=choices_token_ids,
                    expected_label=str(expected_label),
                    other_label=str(other_label),
                    scored_positions=scored_positions,
                    require_binary_labels=True,
                    require_equal_candidate_counts=True,
                )
                return bool(app.static_applicable), str(app.reason)

            primary_a_to_b, reason_a_to_b = _direction_primary(str(dict(it["a"]).get("expected_label", "")))
            primary_b_to_a, reason_b_to_a = _direction_primary(str(dict(it["b"]).get("expected_label", "")))
            primary_static_pair = bool(primary_a_to_b and primary_b_to_a)

            out[pair_id] = {
                "all_single_token": bool(n_multi == 0),
                "n_candidates_total": int(len(token_lens)),
                "n_multi_candidates": int(n_multi),
                "primary_static_pair": bool(primary_static_pair),
                "primary_static_a_to_b": bool(primary_a_to_b),
                "primary_static_b_to_a": bool(primary_b_to_a),
                "primary_reason_a_to_b": str(reason_a_to_b),
                "primary_reason_b_to_a": str(reason_b_to_a),
            }
    if not out:
        raise ValueError(f"No DISAMB rows loaded from {disamb_path}")

    resolved_name = str(getattr(tok, "name_or_path", tokenizer_name_or_path))
    tokenizer_path = Path(resolved_name)
    file_hashes: Dict[str, str] = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
    ):
        fp = tokenizer_path / name
        if fp.exists() and fp.is_file():
            file_hashes[name] = _sha256_file(fp)
    tokenizer_provenance = {
        "tokenizer_name_or_path_input": str(tokenizer_name_or_path),
        "tokenizer_name_or_path_resolved": resolved_name,
        "tokenizer_path_exists": bool(tokenizer_path.exists()),
        "tokenizer_file_hashes": file_hashes,
    }
    return out, tokenizer_provenance


def _build_pair_effects(rows: Iterable[Mapping[str, str]]) -> List[PairEffect]:
    pair_to_a: Dict[str, List[float]] = defaultdict(list)
    pair_to_c: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        pair_id = str(r["pair_id"])
        pair_to_a[pair_id].append(float(r["effect_A"]))
        pair_to_c[pair_id].append(float(r["effect_C"]))

    effects: List[PairEffect] = []
    for pair_id in sorted(pair_to_a.keys()):
        a_vals = pair_to_a[pair_id]
        c_vals = pair_to_c[pair_id]
        if not a_vals or not c_vals:
            continue
        effects.append(
            PairEffect(
                pair_id=pair_id,
                effect_a=float(sum(a_vals) / len(a_vals)),
                effect_c=float(sum(c_vals) / len(c_vals)),
            )
        )
    return effects


def _bootstrap_metrics(
    pairs: Sequence[PairEffect],
    *,
    bootstrap_n: int,
    ci: float,
    seed: int,
    ratio_den_eps: float,
) -> Dict[str, object]:
    if not pairs:
        return {
            "n_pairs": 0,
            "effect_a": {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")},
            "effect_c": {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")},
            "ddm": {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")},
            "crr": {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")},
            "p_ddm_gt_0": float("nan"),
            "p_crr_gt_1": float("nan"),
        }

    point_a = float(sum(p.effect_a for p in pairs) / len(pairs))
    point_c = float(sum(p.effect_c for p in pairs) / len(pairs))
    point_ddm = float(point_c - point_a)
    point_crr = float("nan") if abs(point_a) <= float(ratio_den_eps) else float(point_c / point_a)

    rng = random.Random(int(seed))
    boot_a: List[float] = []
    boot_c: List[float] = []
    boot_ddm: List[float] = []
    boot_crr: List[float] = []
    for _ in range(int(bootstrap_n)):
        sampled = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        m_a = float(sum(p.effect_a for p in sampled) / len(sampled))
        m_c = float(sum(p.effect_c for p in sampled) / len(sampled))
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
    p_ddm_gt_0 = float(sum(1 for x in boot_ddm if x > 0.0) / len(boot_ddm))
    p_crr_gt_1 = float(sum(1 for x in boot_crr if x > 1.0) / len(boot_crr)) if boot_crr else float("nan")

    return {
        "n_pairs": int(len(pairs)),
        "effect_a": {"mean": point_a, "ci_low": a_lo, "ci_high": a_hi},
        "effect_c": {"mean": point_c, "ci_low": c_lo, "ci_high": c_hi},
        "ddm": {"mean": point_ddm, "ci_low": ddm_lo, "ci_high": ddm_hi},
        "crr": {"mean": point_crr, "ci_low": crr_lo, "ci_high": crr_hi},
        "p_ddm_gt_0": p_ddm_gt_0,
        "p_crr_gt_1": p_crr_gt_1,
    }


def _diag_logz_cancellation(rows: Sequence[Mapping[str, str]]) -> Dict[str, float]:
    diffs_a: List[float] = []
    diffs_c: List[float] = []
    for r in rows:
        diffs_a.append(abs(float(r["decomp_exp_delta_logz_mean_A"]) - float(r["decomp_other_delta_logz_mean_A"])))
        diffs_c.append(abs(float(r["decomp_exp_delta_logz_mean_C"]) - float(r["decomp_other_delta_logz_mean_C"])))
    return {
        "max_abs_exp_minus_other_logz_A": float(max(diffs_a) if diffs_a else float("nan")),
        "mean_abs_exp_minus_other_logz_A": float(sum(diffs_a) / len(diffs_a)) if diffs_a else float("nan"),
        "max_abs_exp_minus_other_logz_C": float(max(diffs_c) if diffs_c else float("nan")),
        "mean_abs_exp_minus_other_logz_C": float(sum(diffs_c) / len(diffs_c)) if diffs_c else float("nan"),
    }


def _render_md(
    *,
    out: Mapping[str, object],
) -> str:
    lines: List[str] = []
    lines.append("# MoM Endpoint Plan: Phase 1 Baseline Lock Report")
    lines.append("")
    lines.append("Primary inferential object: paired difference `ΔΔm = Δm_C - Δm_A` (pair-cluster bootstrap).")
    lines.append("")

    src = out["source"]
    lines.append("## Source Artifacts")
    lines.append(f"- Comparability CSV: `{src['comparability_csv']}`")
    lines.append(f"- Comparability summary: `{src['comparability_summary']}`")
    lines.append(f"- DISAMB path: `{src['disamb_path']}`")
    lines.append(f"- Tokenizer: `{src['tokenizer_name_or_path']}`")
    lines.append(f"- Tokenizer resolved path: `{src['tokenizer_name_or_path_resolved']}`")
    lines.append("")

    h = out["hashes"]
    lines.append("## Hashes")
    lines.append(f"- comparability_csv_sha256: `{h['comparability_csv_sha256']}`")
    lines.append(f"- comparability_summary_sha256: `{h['comparability_summary_sha256']}`")
    lines.append(f"- disamb_sha256: `{h['disamb_sha256']}`")
    lines.append(f"- git_commit: `{h['git_commit']}`")
    tok_hashes = dict(h.get("tokenizer_file_hashes", {}))
    if tok_hashes:
        lines.append(f"- tokenizer_file_hashes: `{json.dumps(tok_hashes, sort_keys=True)}`")
    else:
        lines.append("- tokenizer_file_hashes: `none`")
    lines.append("")

    regime = out["regime"]
    lines.append("## Regime Split")
    lines.append(f"- n_pairs_total: `{regime['n_pairs_total']}`")
    lines.append(f"- n_pairs_primary_applicable: `{regime['n_pairs_primary_applicable']}`")
    lines.append(f"- n_pairs_non_primary_applicable: `{regime['n_pairs_non_primary_applicable']}`")
    lines.append(
        f"- non_primary_applicable_pair_ids: `{', '.join(regime['non_primary_applicable_pair_ids']) if regime['non_primary_applicable_pair_ids'] else 'none'}`"
    )
    lines.append(f"- n_pairs_all_single_token: `{regime['n_pairs_all_single_token']}`")
    lines.append("")

    diag = out["diag_logz_cancellation"]
    lines.append("## Diagnostic logZ Cancellation Check")
    lines.append(
        f"- max |ΔlogZ_exp - ΔlogZ_other| (A): `{_fmt(float(diag['max_abs_exp_minus_other_logz_A']), 8)}`"
    )
    lines.append(
        f"- max |ΔlogZ_exp - ΔlogZ_other| (C): `{_fmt(float(diag['max_abs_exp_minus_other_logz_C']), 8)}`"
    )
    lines.append("")

    lines.append("## Layered Results")
    for layer_key in sorted(out["results_by_layer"].keys(), key=lambda x: int(x)):
        layer = out["results_by_layer"][layer_key]
        lines.append(f"### Layer {layer_key}")
        lines.append("")
        lines.append(
            "| split | n_pairs | effect_A mean [CI] | effect_C mean [CI] | ΔΔm mean [CI] | P(ΔΔm>0) | CRR mean [CI] | P(CRR>1) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for split_name in ("all_pairs", "primary_applicable_pairs", "non_primary_applicable_pairs"):
            s = layer[split_name]
            ea = s["effect_a"]
            ec = s["effect_c"]
            ddm = s["ddm"]
            crr = s["crr"]
            lines.append(
                "| {split} | {n} | {ea} [{ealo}, {eahi}] | {ec} [{eclo}, {echi}] | {ddm} [{ddmlo}, {ddmhi}] | {pddm} | {crr} [{crrlo}, {crrhi}] | {pcrr} |".format(
                    split=split_name,
                    n=int(s["n_pairs"]),
                    ea=_fmt(float(ea["mean"])),
                    ealo=_fmt(float(ea["ci_low"])),
                    eahi=_fmt(float(ea["ci_high"])),
                    ec=_fmt(float(ec["mean"])),
                    eclo=_fmt(float(ec["ci_low"])),
                    echi=_fmt(float(ec["ci_high"])),
                    ddm=_fmt(float(ddm["mean"])),
                    ddmlo=_fmt(float(ddm["ci_low"])),
                    ddmhi=_fmt(float(ddm["ci_high"])),
                    pddm=_fmt(float(s["p_ddm_gt_0"]), 3),
                    crr=_fmt(float(crr["mean"])),
                    crrlo=_fmt(float(crr["ci_low"])),
                    crrhi=_fmt(float(crr["ci_high"])),
                    pcrr=_fmt(float(s["p_crr_gt_1"]), 3),
                )
            )
        lines.append("")

    lines.append("## Primary Mechanistic Regime")
    lines.append("- Primary analysis set for mechanism in R1: `primary_applicable_pairs`.")
    lines.append("- `all_pairs` is robustness.")
    lines.append("- `non_primary_applicable_pairs` is descriptive bridge only.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    lock_dir = root / "results" / "paper_lock_20260226T042208Z"
    out_dir = root / "results" / "mom_endpoint_plan"
    p = argparse.ArgumentParser(description="Phase-1 baseline lock report for MoM endpoint plan.")
    p.add_argument("--comparability_csv", type=Path, default=lock_dir / "clt_raw_comparability_l4_l8_l12.csv")
    p.add_argument("--comparability_summary", type=Path, default=lock_dir / "clt_raw_comparability_l4_l8_l12.summary.json")
    p.add_argument("--disamb_path", type=Path, default=root / "data" / "disamb_pairs.jsonl")
    p.add_argument("--tokenizer_name_or_path", type=str, default="")
    p.add_argument("--bootstrap_n", type=int, default=5000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratio_den_eps", type=float, default=1e-8)
    p.add_argument("--out_json", type=Path, default=out_dir / "baseline_metrics.json")
    p.add_argument("--out_md", type=Path, default=out_dir / "baseline_report.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer_name = _infer_tokenizer_name(args.comparability_summary, args.tokenizer_name_or_path)
    rows = _load_analysis_rows(args.comparability_csv)
    pair_profile, tokenizer_provenance = _pair_primary_profile(args.disamb_path, tokenizer_name)

    missing_pairs = sorted({str(r["pair_id"]) for r in rows} - set(pair_profile.keys()))
    if missing_pairs:
        raise ValueError(f"Missing pair IDs in dataset for analysis rows: {missing_pairs[:5]}")

    layers = sorted({int(r["layer"]) for r in rows})
    results_by_layer: Dict[str, Dict[str, object]] = {}
    for layer in layers:
        layer_rows = [r for r in rows if int(r["layer"]) == int(layer)]
        all_effects = _build_pair_effects(layer_rows)
        primary_effects = [p for p in all_effects if bool(pair_profile[p.pair_id]["primary_static_pair"])]
        non_primary_effects = [p for p in all_effects if not bool(pair_profile[p.pair_id]["primary_static_pair"])]
        results_by_layer[str(int(layer))] = {
            "all_pairs": _bootstrap_metrics(
                all_effects,
                bootstrap_n=int(args.bootstrap_n),
                ci=float(args.ci),
                seed=int(args.seed),
                ratio_den_eps=float(args.ratio_den_eps),
            ),
            "primary_applicable_pairs": _bootstrap_metrics(
                primary_effects,
                bootstrap_n=int(args.bootstrap_n),
                ci=float(args.ci),
                seed=int(args.seed),
                ratio_den_eps=float(args.ratio_den_eps),
            ),
            "non_primary_applicable_pairs": _bootstrap_metrics(
                non_primary_effects,
                bootstrap_n=int(args.bootstrap_n),
                ci=float(args.ci),
                seed=int(args.seed),
                ratio_den_eps=float(args.ratio_den_eps),
            ),
        }

    repo_root = Path(__file__).resolve().parents[1]
    try:
        git_head = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_head = "unknown"

    out: Dict[str, object] = {
        "source": {
            "comparability_csv": str(args.comparability_csv),
            "comparability_summary": str(args.comparability_summary),
            "disamb_path": str(args.disamb_path),
            "tokenizer_name_or_path": tokenizer_name,
            "tokenizer_name_or_path_resolved": str(tokenizer_provenance["tokenizer_name_or_path_resolved"]),
        },
        "hashes": {
            "comparability_csv_sha256": _sha256_file(args.comparability_csv),
            "comparability_summary_sha256": _sha256_file(args.comparability_summary),
            "disamb_sha256": _sha256_file(args.disamb_path),
            "git_commit": git_head,
            "tokenizer_file_hashes": tokenizer_provenance["tokenizer_file_hashes"],
        },
        "config": {
            "bootstrap_n": int(args.bootstrap_n),
            "ci": float(args.ci),
            "seed": int(args.seed),
            "ratio_den_eps": float(args.ratio_den_eps),
        },
        "regime": {
            "n_pairs_total": int(len(pair_profile)),
            "n_pairs_primary_applicable": int(sum(1 for p in pair_profile.values() if bool(p["primary_static_pair"]))),
            "n_pairs_non_primary_applicable": int(
                sum(1 for p in pair_profile.values() if not bool(p["primary_static_pair"]))
            ),
            "non_primary_applicable_pair_ids": sorted(
                [pid for pid, prof in pair_profile.items() if not bool(prof["primary_static_pair"])]
            ),
            "n_pairs_all_single_token": int(sum(1 for p in pair_profile.values() if bool(p["all_single_token"]))),
        },
        "diag_logz_cancellation": _diag_logz_cancellation(rows),
        "results_by_layer": results_by_layer,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_md(out=out), encoding="utf-8")
    print(str(args.out_json))
    print(str(args.out_md))


if __name__ == "__main__":
    main()
