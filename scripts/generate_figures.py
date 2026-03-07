from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _pick_representative(rows: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Pick a single representative row for a model.

    AoM evaluation is deterministic for log-prob scoring; multiple seed rows are typically identical.
    Prefer the smallest seed if present, else the first row.
    """
    if not rows:
        return {}
    rows_sorted = sorted(rows, key=lambda r: int(r.get("seed", "0") or 0))
    return rows_sorted[0]


def _err_from_ci(mu: float, lo: float, hi: float) -> Tuple[float, float]:
    return max(0.0, mu - lo), max(0.0, hi - mu)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="CSV produced by aom_eval.py")
    p.add_argument("--out_dir", type=str, default="figures")
    p.add_argument("--fmt", type=str, default="png", choices=["png", "pdf"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    with open(args.input, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    by_model_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_model_rows[r.get("model", "unknown")].append(r)

    models = sorted(by_model_rows.keys())

    # Composite (point estimate; CI requires joint bootstrap across metrics).
    fig, ax = plt.subplots(figsize=(8, 4))
    composite_mean = [float(_pick_representative(by_model_rows[m]).get("aom_composite", 0.0) or 0.0) for m in models]
    ax.bar(models, composite_mean)
    ax.set_title("AoM composite by model")
    ax.set_ylabel("AoM composite (point estimate)")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"aom_composite.{args.fmt}")
    plt.close(fig)

    # Breakdown plot
    metrics = [
        ("disamb_accuracy", "AoM-DISAMB"),
        ("cf_shift_direction_accuracy", "AoM-CF"),
        ("coh_constraint_accuracy", "AoM-COH"),
    ]
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, (key, label) in enumerate(metrics):
        means: List[float] = []
        err_lo: List[float] = []
        err_hi: List[float] = []
        for m in models:
            r = _pick_representative(by_model_rows[m])
            mu = float(r.get(key, 0.0) or 0.0)
            lo = float(r.get(f"{key}_ci_low", 0.0) or 0.0)
            hi = float(r.get(f"{key}_ci_high", 0.0) or 0.0)
            e_lo, e_hi = _err_from_ci(mu, lo, hi)
            means.append(mu)
            err_lo.append(e_lo)
            err_hi.append(e_hi)
        yerr = np.array([err_lo, err_hi])
        ax.bar(x + (j - 1) * width, means, width, yerr=yerr, capsize=3, label=label)
    ax.set_xticks(x, models, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("AoM task breakdown (bootstrap CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"aom_breakdown.{args.fmt}")
    plt.close(fig)

    # DISAMB keyword baseline comparison (if present).
    if any("baseline_disamb_keyword_accuracy" in r for r in rows):
        fig, ax = plt.subplots(figsize=(9, 4))
        xs = np.arange(len(models))
        model_mu = []
        model_err_lo = []
        model_err_hi = []
        base_mu = []
        base_err_lo = []
        base_err_hi = []
        for m in models:
            r = _pick_representative(by_model_rows[m])
            mu = float(r.get("disamb_accuracy", 0.0) or 0.0)
            lo = float(r.get("disamb_accuracy_ci_low", 0.0) or 0.0)
            hi = float(r.get("disamb_accuracy_ci_high", 0.0) or 0.0)
            e_lo, e_hi = _err_from_ci(mu, lo, hi)
            model_mu.append(mu)
            model_err_lo.append(e_lo)
            model_err_hi.append(e_hi)

            mu = float(r.get("baseline_disamb_keyword_accuracy", 0.0) or 0.0)
            lo = float(r.get("baseline_disamb_keyword_accuracy_ci_low", 0.0) or 0.0)
            hi = float(r.get("baseline_disamb_keyword_accuracy_ci_high", 0.0) or 0.0)
            e_lo, e_hi = _err_from_ci(mu, lo, hi)
            base_mu.append(mu)
            base_err_lo.append(e_lo)
            base_err_hi.append(e_hi)

        w = 0.35
        ax.bar(xs - w / 2, model_mu, w, yerr=np.array([model_err_lo, model_err_hi]), capsize=3, label="model")
        ax.bar(xs + w / 2, base_mu, w, yerr=np.array([base_err_lo, base_err_hi]), capsize=3, label="keyword baseline")
        ax.set_xticks(xs, models, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("AoM-DISAMB vs keyword baseline (bootstrap CI)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"disamb_vs_baseline.{args.fmt}")
        plt.close(fig)

    # CF sham controls: absolute shift magnitude for shift vs invariant items (if present).
    if any("cf_mean_abs_shift_pref_invariant_items" in r for r in rows):
        fig, ax = plt.subplots(figsize=(9, 4))
        xs = np.arange(len(models))
        shift_mu = []
        shift_err_lo = []
        shift_err_hi = []
        sham_mu = []
        sham_err_lo = []
        sham_err_hi = []
        for m in models:
            r = _pick_representative(by_model_rows[m])

            mu = float(r.get("cf_mean_abs_shift_pref_shift_items", 0.0) or 0.0)
            lo = float(r.get("cf_mean_abs_shift_pref_shift_items_ci_low", 0.0) or 0.0)
            hi = float(r.get("cf_mean_abs_shift_pref_shift_items_ci_high", 0.0) or 0.0)
            e_lo, e_hi = _err_from_ci(mu, lo, hi)
            shift_mu.append(mu)
            shift_err_lo.append(e_lo)
            shift_err_hi.append(e_hi)

            mu = float(r.get("cf_mean_abs_shift_pref_invariant_items", 0.0) or 0.0)
            lo = float(r.get("cf_mean_abs_shift_pref_invariant_items_ci_low", 0.0) or 0.0)
            hi = float(r.get("cf_mean_abs_shift_pref_invariant_items_ci_high", 0.0) or 0.0)
            e_lo, e_hi = _err_from_ci(mu, lo, hi)
            sham_mu.append(mu)
            sham_err_lo.append(e_lo)
            sham_err_hi.append(e_hi)

        w = 0.35
        ax.bar(xs - w / 2, shift_mu, w, yerr=np.array([shift_err_lo, shift_err_hi]), capsize=3, label="shift items")
        ax.bar(xs + w / 2, sham_mu, w, yerr=np.array([sham_err_lo, sham_err_hi]), capsize=3, label="sham items")
        ax.set_xticks(xs, models, rotation=30, ha="right")
        ax.set_title("AoM-CF shift magnitude controls (bootstrap CI)")
        ax.set_ylabel("Mean |Δ_shift|")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"cf_sham_controls.{args.fmt}")
        plt.close(fig)

    # COH ablation controls (if present).
    g_rel = "coh_group_ablate_relevant_constraint_accuracy"
    g_irr = "coh_group_ablate_irrelevant_constraint_accuracy"
    if any((g_rel in r and g_irr in r) for r in rows):
        fig, ax = plt.subplots(figsize=(9, 4))
        xs = np.arange(len(models))
        rel_mu = []
        rel_err_lo = []
        rel_err_hi = []
        irr_mu = []
        irr_err_lo = []
        irr_err_hi = []
        for m in models:
            r = _pick_representative(by_model_rows[m])

            mu = float(r.get(g_rel, 0.0) or 0.0)
            lo = float(r.get(f"{g_rel}_ci_low", 0.0) or 0.0)
            hi = float(r.get(f"{g_rel}_ci_high", 0.0) or 0.0)
            e_lo, e_hi = _err_from_ci(mu, lo, hi)
            rel_mu.append(mu)
            rel_err_lo.append(e_lo)
            rel_err_hi.append(e_hi)

            mu = float(r.get(g_irr, 0.0) or 0.0)
            lo = float(r.get(f"{g_irr}_ci_low", 0.0) or 0.0)
            hi = float(r.get(f"{g_irr}_ci_high", 0.0) or 0.0)
            e_lo, e_hi = _err_from_ci(mu, lo, hi)
            irr_mu.append(mu)
            irr_err_lo.append(e_lo)
            irr_err_hi.append(e_hi)

        w = 0.35
        ax.bar(xs - w / 2, rel_mu, w, yerr=np.array([rel_err_lo, rel_err_hi]), capsize=3, label="ablate relevant")
        ax.bar(xs + w / 2, irr_mu, w, yerr=np.array([irr_err_lo, irr_err_hi]), capsize=3, label="ablate irrelevant")
        ax.set_xticks(xs, models, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("AoM-COH ablation controls (bootstrap CI)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"coh_ablations.{args.fmt}")
        plt.close(fig)

    # COH paired deltas (preferred analysis if triplets are present).
    d_rel = "coh_paired_delta_ablate_relevant_accuracy"
    d_irr = "coh_paired_delta_ablate_irrelevant_accuracy"
    if any((d_rel in r and d_irr in r) for r in rows):
        fig, ax = plt.subplots(figsize=(9, 4))
        xs = np.arange(len(models))
        rel_mu = []
        rel_err_lo = []
        rel_err_hi = []
        irr_mu = []
        irr_err_lo = []
        irr_err_hi = []
        for m in models:
            r = _pick_representative(by_model_rows[m])

            mu = float(r.get(d_rel, 0.0) or 0.0)
            lo = float(r.get(f"{d_rel}_ci_low", 0.0) or 0.0)
            hi = float(r.get(f"{d_rel}_ci_high", 0.0) or 0.0)
            e_lo, e_hi = _err_from_ci(mu, lo, hi)
            rel_mu.append(mu)
            rel_err_lo.append(e_lo)
            rel_err_hi.append(e_hi)

            mu = float(r.get(d_irr, 0.0) or 0.0)
            lo = float(r.get(f"{d_irr}_ci_low", 0.0) or 0.0)
            hi = float(r.get(f"{d_irr}_ci_high", 0.0) or 0.0)
            e_lo, e_hi = _err_from_ci(mu, lo, hi)
            irr_mu.append(mu)
            irr_err_lo.append(e_lo)
            irr_err_hi.append(e_hi)

        w = 0.35
        ax.bar(xs - w / 2, rel_mu, w, yerr=np.array([rel_err_lo, rel_err_hi]), capsize=3, label="main − ablate relevant")
        ax.bar(xs + w / 2, irr_mu, w, yerr=np.array([irr_err_lo, irr_err_hi]), capsize=3, label="main − ablate irrelevant")
        ax.set_xticks(xs, models, rotation=30, ha="right")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title("AoM-COH paired deltas (bootstrap CI over base_id)")
        ax.set_ylabel("Δ accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"coh_paired_deltas.{args.fmt}")
        plt.close(fig)

    # CPT layer sweep (if present)
    # Keys are produced as `cpt_effect_layer_{i}` / `cpt_sham_effect_layer_{i}`.
    any_cpt = any(any(k.startswith("cpt_effect_layer_") for k in r.keys()) for r in rows)
    if any_cpt:
        for model in models:
            layer_to_vals: Dict[int, List[float]] = defaultdict(list)
            layer_to_sham: Dict[int, List[float]] = defaultdict(list)
            for r in rows:
                if r.get("model", "unknown") != model:
                    continue
                for k, v in r.items():
                    if not v:
                        continue
                    if k.startswith("cpt_effect_layer_"):
                        layer = int(k.split("_")[-1])
                        layer_to_vals[layer].append(float(v))
                    if k.startswith("cpt_sham_effect_layer_"):
                        layer = int(k.split("_")[-1])
                        layer_to_sham[layer].append(float(v))

            layers_sorted = sorted(layer_to_vals.keys())
            if not layers_sorted:
                continue

            eff_mean = []
            eff_std = []
            sham_mean = []
            sham_std = []
            for l in layers_sorted:
                vals = layer_to_vals.get(l, [])
                mu = float(np.mean(vals)) if vals else 0.0
                sd = float(np.std(vals, ddof=0)) if vals else 0.0
                eff_mean.append(mu)
                eff_std.append(sd)
                vals = layer_to_sham.get(l, [])
                mu = float(np.mean(vals)) if vals else 0.0
                sd = float(np.std(vals, ddof=0)) if vals else 0.0
                sham_mean.append(mu)
                sham_std.append(sd)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.errorbar(layers_sorted, eff_mean, yerr=eff_std, marker="o", label="patched")
            ax.errorbar(layers_sorted, sham_mean, yerr=sham_std, marker="o", linestyle="--", label="sham")
            ax.axhline(0.0, color="black", linewidth=1)
            ax.set_title(f"CPT patching effect by layer ({model})")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Mean effect (patched margin − base margin)")
            ax.legend()
            fig.tight_layout()
            safe_name = model.replace("/", "_")
            fig.savefig(out_dir / f"cpt_layer_effect_{safe_name}.{args.fmt}")
            plt.close(fig)


if __name__ == "__main__":
    main()
