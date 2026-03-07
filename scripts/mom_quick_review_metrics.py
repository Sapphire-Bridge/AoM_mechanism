from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Mapping, Sequence, Tuple

from transformers import AutoTokenizer


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs)
    den_y = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(den_x * den_y)
    if den <= 0.0:
        return float("nan")
    return float(num / den)


def _pair_means(rows: Sequence[Mapping[str, str]], key: str) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(float(row[key]))
    return {pid: float(sum(vals) / len(vals)) for pid, vals in grouped.items() if vals}


def _bootstrap_percentile(values: Sequence[float], ci: float = 0.95) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not arr:
        return float("nan"), float("nan"), float("nan")
    alpha = max(0.0, min(1.0, float(ci)))
    lo_i = int(((1.0 - alpha) / 2.0) * len(arr))
    hi_i = max(lo_i, int(((1.0 + alpha) / 2.0) * len(arr)) - 1)
    lo_i = min(max(lo_i, 0), len(arr) - 1)
    hi_i = min(max(hi_i, 0), len(arr) - 1)
    mean_val = float(sum(arr) / len(arr))
    return mean_val, float(arr[lo_i]), float(arr[hi_i])


def _read_layer_rows(path: Path, layer: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["layer"]) != int(layer):
                continue
            if str(row.get("analysis_included", "False")) != "True":
                continue
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"No analysis-included rows found for layer={layer} in {path}")
    return rows


def _compute_layer4_metrics(
    *,
    rows: Sequence[Mapping[str, str]],
    bootstrap_n: int,
    seed: int,
) -> Dict[str, object]:
    pair_ids = sorted({str(r["pair_id"]) for r in rows})
    if not pair_ids:
        raise ValueError("No pair IDs found in layer rows")

    pm: Dict[str, Dict[str, float]] = {
        "effect_A": _pair_means(rows, "effect_A"),
        "effect_C": _pair_means(rows, "effect_C"),
        "kl_A": _pair_means(rows, "decomp_exp_kl_base_to_patch_mean_A"),
        "kl_C": _pair_means(rows, "decomp_exp_kl_base_to_patch_mean_C"),
        "rms_A": _pair_means(rows, "decomp_exp_rms_logit_change_mean_A"),
        "rms_C": _pair_means(rows, "decomp_exp_rms_logit_change_mean_C"),
        "diag_A": _pair_means(rows, "decomp_exp_delta_logprob_target_mean_A"),
        "diag_C": _pair_means(rows, "decomp_exp_delta_logprob_target_mean_C"),
        "d_CA": _pair_means(rows, "d_CA"),
    }

    rng = random.Random(int(seed))
    boot: Dict[str, List[float]] = defaultdict(list)
    for _ in range(int(bootstrap_n)):
        sampled = [pair_ids[rng.randrange(len(pair_ids))] for _ in range(len(pair_ids))]

        def _mean(name: str) -> float:
            vals = [pm[name][pid] for pid in sampled]
            return float(sum(vals) / len(vals))

        m_effect_a = _mean("effect_A")
        m_effect_c = _mean("effect_C")
        m_kl_a = _mean("kl_A")
        m_kl_c = _mean("kl_C")
        m_rms_a = _mean("rms_A")
        m_rms_c = _mean("rms_C")

        eff_kl_a = m_effect_a / m_kl_a if abs(m_kl_a) > 0.0 else float("nan")
        eff_kl_c = m_effect_c / m_kl_c if abs(m_kl_c) > 0.0 else float("nan")
        eff_rms_a = m_effect_a / m_rms_a if abs(m_rms_a) > 0.0 else float("nan")
        eff_rms_c = m_effect_c / m_rms_c if abs(m_rms_c) > 0.0 else float("nan")

        boot["effect_A"].append(m_effect_a)
        boot["effect_C"].append(m_effect_c)
        boot["kl_A"].append(m_kl_a)
        boot["kl_C"].append(m_kl_c)
        boot["rms_A"].append(m_rms_a)
        boot["rms_C"].append(m_rms_c)
        boot["eff_kl_A"].append(eff_kl_a)
        boot["eff_kl_C"].append(eff_kl_c)
        boot["eff_rms_A"].append(eff_rms_a)
        boot["eff_rms_C"].append(eff_rms_c)
        boot["eff_ratio_kl"].append(eff_kl_c / eff_kl_a if abs(eff_kl_a) > 0.0 else float("nan"))
        boot["eff_ratio_rms"].append(eff_rms_c / eff_rms_a if abs(eff_rms_a) > 0.0 else float("nan"))

        xs_a = [pm["effect_A"][pid] for pid in sampled]
        ys_a = [pm["diag_A"][pid] for pid in sampled]
        xs_c = [pm["effect_C"][pid] for pid in sampled]
        ys_c = [pm["diag_C"][pid] for pid in sampled]
        xs_delta = [pm["d_CA"][pid] for pid in sampled]
        ys_delta = [pm["diag_C"][pid] - pm["diag_A"][pid] for pid in sampled]
        boot["corr_effect_vs_diag_A"].append(_pearson(xs_a, ys_a))
        boot["corr_effect_vs_diag_C"].append(_pearson(xs_c, ys_c))
        boot["corr_dCA_vs_ddiag"].append(_pearson(xs_delta, ys_delta))

    def _metric(name: str) -> Dict[str, float]:
        mean_v, lo, hi = _bootstrap_percentile(boot[name], ci=0.95)
        return {"mean": mean_v, "ci_low": lo, "ci_high": hi}

    return {
        "n_rows": int(len(rows)),
        "n_pairs": int(len(pair_ids)),
        "effect": {"raw_A": _metric("effect_A"), "sae_C": _metric("effect_C")},
        "disturbance": {
            "kl_base_to_patch_exp": {"raw_A": _metric("kl_A"), "sae_C": _metric("kl_C")},
            "rms_logit_change_exp": {"raw_A": _metric("rms_A"), "sae_C": _metric("rms_C")},
        },
        "efficiency": {
            "delta_m_per_kl": {"raw_A": _metric("eff_kl_A"), "sae_C": _metric("eff_kl_C"), "sae_over_raw": _metric("eff_ratio_kl")},
            "delta_m_per_rms": {"raw_A": _metric("eff_rms_A"), "sae_C": _metric("eff_rms_C"), "sae_over_raw": _metric("eff_ratio_rms")},
        },
        "correlation": {
            "pair_pearson_effect_vs_diag_raw_A": _metric("corr_effect_vs_diag_A"),
            "pair_pearson_effect_vs_diag_sae_C": _metric("corr_effect_vs_diag_C"),
            "pair_pearson_dCA_vs_diag_delta": _metric("corr_dCA_vs_ddiag"),
        },
    }


def _candidate_stats(disamb_path: Path, tokenizer_name_or_path: str) -> Dict[str, object]:
    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, local_files_only=True)

    items: List[Dict[str, object]] = []
    with disamb_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    cand_counts: List[int] = []
    token_lens: List[int] = []
    n_label_sets_with_shared_first = 0
    n_label_sets_total = 0
    for it in items:
        choices = dict(it["choices"])
        for cands in choices.values():
            cands_list = list(cands)
            cand_counts.append(len(cands_list))
            first_tokens: List[int | None] = []
            for cont in cands_list:
                ids = tok.encode(str(cont), add_special_tokens=False)
                token_lens.append(len(ids))
                first_tokens.append(ids[0] if ids else None)
            shared = False
            for i in range(len(first_tokens)):
                for j in range(i + 1, len(first_tokens)):
                    if first_tokens[i] is not None and first_tokens[i] == first_tokens[j]:
                        shared = True
            n_label_sets_total += 1
            if shared:
                n_label_sets_with_shared_first += 1

    n_candidates_total = len(token_lens)
    len_counts: Dict[str, int] = {}
    for ln in sorted(set(token_lens)):
        len_counts[str(int(ln))] = int(sum(1 for x in token_lens if int(x) == int(ln)))

    n_single = int(sum(1 for x in token_lens if int(x) == 1))
    n_multi = int(n_candidates_total - n_single)

    return {
        "n_items": int(len(items)),
        "n_label_sets": int(n_label_sets_total),
        "candidates_per_label_unique": sorted({int(x) for x in cand_counts}),
        "candidates_per_label_mean": float(sum(cand_counts) / len(cand_counts)),
        "n_candidates_total": int(n_candidates_total),
        "token_length": {
            "mean": float(sum(token_lens) / len(token_lens)),
            "median": float(median(token_lens)),
            "min": int(min(token_lens)),
            "max": int(max(token_lens)),
            "counts": len_counts,
            "single_token_frac": float(n_single / n_candidates_total),
            "multi_token_frac": float(n_multi / n_candidates_total),
        },
        "shared_first_token_within_label_set_frac": float(
            n_label_sets_with_shared_first / max(1, n_label_sets_total)
        ),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    paper_lock = root / "results" / "paper_lock_20260226T042208Z"
    parser = argparse.ArgumentParser(description="Compute quick-review MoM metrics from existing artifacts.")
    parser.add_argument(
        "--comparability_csv",
        type=Path,
        default=paper_lock / "clt_raw_comparability_l4_l8_l12.csv",
    )
    parser.add_argument(
        "--comparability_summary",
        type=Path,
        default=paper_lock / "clt_raw_comparability_l4_l8_l12.summary.json",
    )
    parser.add_argument("--disamb_path", type=Path, default=root / "data" / "disamb_pairs.jsonl")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--bootstrap_n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_json",
        type=Path,
        default=paper_lock / "mom_quick_review_metrics.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.comparability_summary.open() as f:
        summary = json.load(f)
    tok_name_or_path = str(args.tokenizer_name_or_path).strip() or str(summary["model_name_or_path"])

    rows = _read_layer_rows(args.comparability_csv, layer=int(args.layer))
    layer_metrics = _compute_layer4_metrics(rows=rows, bootstrap_n=int(args.bootstrap_n), seed=int(args.seed))
    cand_stats = _candidate_stats(args.disamb_path, tok_name_or_path)

    out = {
        "source_artifacts": {
            "comparability_csv": str(args.comparability_csv),
            "comparability_summary": str(args.comparability_summary),
            "disamb_path": str(args.disamb_path),
            "tokenizer_name_or_path": tok_name_or_path,
        },
        "config": {
            "layer": int(args.layer),
            "bootstrap_n": int(args.bootstrap_n),
            "seed": int(args.seed),
        },
        "layer_metrics": layer_metrics,
        "candidate_stats": cand_stats,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(str(args.out_json))


if __name__ == "__main__":
    main()
