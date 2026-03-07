from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_first_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
    if row is None:
        raise ValueError(f"No rows found in CSV: {path}")
    return row


def _load_layer_summary(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_layer = payload.get("per_layer", [])
    if not isinstance(per_layer, list):
        raise ValueError(f"Expected list at per_layer in {path}")
    out: dict[int, dict[str, Any]] = {}
    for row in per_layer:
        layer = int(row["layer"])
        out[layer] = row
    return out


def _ci_err(mean: float, ci_low: float, ci_high: float) -> tuple[float, float]:
    return float(max(0.0, mean - ci_low)), float(max(0.0, ci_high - mean))


def _make_figure_1(
    *,
    raw_csv: Path,
    sae_csv: Path,
    controls_summary: Path,
    out_path: Path,
) -> None:
    raw_row = _load_first_row(raw_csv)
    sae_row = _load_first_row(sae_csv)
    layer_summary = _load_layer_summary(controls_summary)

    layers = [4, 8, 12, 16, 20, 24]
    raw_means = np.array([float(raw_row[f"cpt_effect_layer_{layer}"]) for layer in layers], dtype=float)
    sae_means = np.array([float(sae_row[f"clt_cpt_effect_layer_{layer}"]) for layer in layers], dtype=float)

    raw_low = np.zeros(len(layers), dtype=float)
    raw_high = np.zeros(len(layers), dtype=float)
    sae_low = np.zeros(len(layers), dtype=float)
    sae_high = np.zeros(len(layers), dtype=float)
    for i, layer in enumerate(layers):
        if layer not in layer_summary:
            continue
        row = layer_summary[layer]
        rl, rh = _ci_err(float(row["effect_A_mean"]), float(row["effect_A_ci_low"]), float(row["effect_A_ci_high"]))
        sl, sh = _ci_err(float(row["effect_C_mean"]), float(row["effect_C_ci_low"]), float(row["effect_C_ci_high"]))
        raw_low[i], raw_high[i] = rl, rh
        sae_low[i], sae_high[i] = sl, sh

    x = np.arange(len(layers), dtype=float)
    width = 0.38

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(
        x - width / 2.0,
        raw_means,
        width=width,
        label="Raw",
        color="#4C78A8",
        yerr=np.vstack([raw_low, raw_high]),
        capsize=3,
    )
    ax.bar(
        x + width / 2.0,
        sae_means,
        width=width,
        label="SAE",
        color="#F58518",
        yerr=np.vstack([sae_low, sae_high]),
        capsize=3,
    )
    ax.set_xticks(x, [str(layer) for layer in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean donor-directed margin effect (Δm)")
    ax.set_title("DISAMB six-layer profile: Raw vs SAE")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def _make_figure_2(*, controls_summary: Path, out_path: Path) -> None:
    layer_summary = _load_layer_summary(controls_summary)
    layers = [4, 8, 12]

    arm_specs = [
        ("Raw A", "effect_A_mean", "effect_A_ci_low", "effect_A_ci_high", "#4C78A8"),
        ("PCA", "effect_PRJ_PCA_mean", "effect_PRJ_PCA_ci_low", "effect_PRJ_PCA_ci_high", "#72B7B2"),
        ("Random", "effect_PRJ_RAND_mean_mean", "effect_PRJ_RAND_mean_ci_low", "effect_PRJ_RAND_mean_ci_high", "#54A24B"),
        ("RECON", "effect_STRESS_RECON_mean", "effect_STRESS_RECON_ci_low", "effect_STRESS_RECON_ci_high", "#F58518"),
        ("RESID", "effect_STRESS_RESID_mean", "effect_STRESS_RESID_ci_low", "effect_STRESS_RESID_ci_high", "#E45756"),
    ]

    x = np.arange(len(layers), dtype=float)
    width = 0.16

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for i, (label, mean_key, lo_key, hi_key, color) in enumerate(arm_specs):
        vals = []
        lows = []
        highs = []
        for layer in layers:
            row = layer_summary[layer]
            mean = float(row[mean_key])
            lo = float(row[lo_key])
            hi = float(row[hi_key])
            err_lo, err_hi = _ci_err(mean, lo, hi)
            vals.append(mean)
            lows.append(err_lo)
            highs.append(err_hi)
        offset = (i - (len(arm_specs) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            np.array(vals, dtype=float),
            width=width,
            label=label,
            color=color,
            yerr=np.vstack([np.array(lows, dtype=float), np.array(highs, dtype=float)]),
            capsize=3,
        )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x, [str(layer) for layer in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean donor-directed margin effect (Δm)")
    ax.set_title("DISAMB matched controls: Raw/PCA/Random/RECON/RESID")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def _make_figure_3(*, controls_summary: Path, out_path: Path) -> None:
    layer_summary = _load_layer_summary(controls_summary)
    layers = [4, 8, 12]
    x = np.array([float(layer_summary[layer]["fidelity_cosine_mean"]) for layer in layers], dtype=float)
    y = np.array([float(layer_summary[layer]["crr_C_over_A_mean"]) for layer in layers], dtype=float)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(x, y, s=90, color="#4C78A8")
    for layer, xi, yi in zip(layers, x, y):
        ax.annotate(f"L{layer}", (xi, yi), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Activation cosine (reconstruction fidelity)")
    ax.set_ylabel("CRR (SAE / Raw)")
    ax.set_title("Fidelity vs recovery (layers 4/8/12)")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MoM paper figure PDFs from existing artifacts.")
    parser.add_argument(
        "--raw_6layer_csv",
        type=Path,
        default=Path("results/overnight_mech_20260225T135850Z/gemma2b_raw_6layer_full_seed42.csv"),
    )
    parser.add_argument(
        "--sae_6layer_csv",
        type=Path,
        default=Path("results/overnight_mech_20260225T135850Z/gemma2b_clt_6layer_full_seed42.csv"),
    )
    parser.add_argument(
        "--controls_summary",
        type=Path,
        default=Path("results/clt_raw_comparability_l4_l8_l12_controls_full.summary.json"),
    )
    parser.add_argument("--out_dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _make_figure_1(
        raw_csv=args.raw_6layer_csv,
        sae_csv=args.sae_6layer_csv,
        controls_summary=args.controls_summary,
        out_path=out_dir / "fig1_disamb_layer_profile.pdf",
    )
    _make_figure_2(
        controls_summary=args.controls_summary,
        out_path=out_dir / "fig2_controls_recon_resid.pdf",
    )
    _make_figure_3(
        controls_summary=args.controls_summary,
        out_path=out_dir / "fig3_fidelity_vs_recovery.pdf",
    )


if __name__ == "__main__":
    main()
