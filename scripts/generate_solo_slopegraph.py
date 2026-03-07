from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


LEFT_COL_DEFAULT = "coh_paired_delta_ablate_irrelevant_mean_gap"
RIGHT_COL_DEFAULT = "coh_paired_delta_ablate_relevant_mean_gap"

# Canonical Graph 3 values provided for the JoLLI solo slopegraph.
DEFAULT_DATA: dict[str, tuple[float, float]] = {
    "GPT-2 (124M)": (0.411, 1.094),
    "Qwen 2.5 (0.5B)": (0.332, 1.687),
    "Qwen 2.5 (1.5B)": (0.463, 1.838),
    "Qwen 2.5 (3B)": (0.336, 1.403),
    "Llama 3.2 (1B)": (0.450, 1.507),
    "Llama 3.2 (3B)": (0.521, 1.288),
}

MODEL_ALIASES: dict[str, str] = {
    "gpt2": "GPT-2 (124M)",
    "openai-community/gpt2": "GPT-2 (124M)",
    "Qwen/Qwen2.5-0.5B": "Qwen 2.5 (0.5B)",
    "Qwen/Qwen2.5-1.5B": "Qwen 2.5 (1.5B)",
    "Qwen/Qwen2.5-3B": "Qwen 2.5 (3B)",
    "meta-llama/Llama-3.2-1B": "Llama 3.2 (1B)",
    "meta-llama/Llama-3.2-3B": "Llama 3.2 (3B)",
}

COLORS = {
    "GPT": "#888888",
    "Qwen": "#77AADD",
    "Llama": "#EE8866",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Graph 3 solo slopegraph (publication-style).")
    p.add_argument(
        "--out",
        type=str,
        default="figures/Fig1_Slopegraph_JoLLI.pdf",
        help="Output figure path (.pdf or .png).",
    )
    p.add_argument(
        "--csv_path",
        type=str,
        default="",
        help="Optional CSV path to populate from file. If omitted, uses canonical values in this script.",
    )
    p.add_argument("--left_col", type=str, default=LEFT_COL_DEFAULT)
    p.add_argument("--right_col", type=str, default=RIGHT_COL_DEFAULT)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _family_color(name: str) -> str:
    if "GPT" in name:
        return COLORS["GPT"]
    if "Qwen" in name:
        return COLORS["Qwen"]
    return COLORS["Llama"]


def _load_from_csv(path: Path, *, left_col: str, right_col: str) -> dict[str, tuple[float, float]]:
    extracted: dict[str, tuple[float, float]] = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            model_name = str(row.get("model", "")).strip()
            display = MODEL_ALIASES.get(model_name, "")
            if not display:
                continue
            left_s = str(row.get(left_col, "")).strip()
            right_s = str(row.get(right_col, "")).strip()
            if not left_s or not right_s:
                continue
            extracted[display] = (float(left_s), float(right_s))

    missing = [k for k in DEFAULT_DATA if k not in extracted]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required models in CSV extraction: {missing_str}")

    # Preserve plotting order.
    return {k: extracted[k] for k in DEFAULT_DATA}


def _plot(data: dict[str, tuple[float, float]], out_path: Path, *, show: bool) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "figure.figsize": (7, 6),
        }
    )

    fig, ax = plt.subplots()

    for name, (left_val, right_val) in data.items():
        color = _family_color(name)
        ax.plot([0, 1], [left_val, right_val], color=color, linewidth=2.5, alpha=0.9, marker="o", markersize=6)
        ax.text(
            -0.02,
            left_val,
            f"{left_val:.2f}",
            ha="right",
            va="center",
            color=color,
            fontweight="bold",
            fontsize=9,
        )
        ax.text(
            1.02,
            right_val,
            f"{right_val:.2f}  {name}",
            ha="left",
            va="center",
            color=color,
            fontweight="bold",
            fontsize=9,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Length Control\n(Irrelevant Ablation)", "Semantic Load\n(Constraint Ablation)"],
        fontweight="bold",
        fontsize=11,
    )
    ax.set_ylabel("Coherence Degradation (Log-Prob Margin Drop)\nHigher = More Damage to Coherence", fontsize=10)
    ax.set_title("The Informational Role of Context", fontweight="bold", pad=20)

    ax.annotate(
        "",
        xy=(0.5, 1.3),
        xytext=(0.5, 0.6),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.5, "ls": "--"},
    )
    ax.text(0.52, 0.95, "Semantic\nImpact", ha="left", va="center", style="italic", fontsize=10)

    ax.grid(axis="x")
    ax.set_xlim(-0.2, 1.4)
    ax.set_ylim(0.0, 2.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format=out_path.suffix.lstrip(".") or "pdf", dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    data = DEFAULT_DATA.copy()
    if args.csv_path:
        data = _load_from_csv(Path(args.csv_path), left_col=args.left_col, right_col=args.right_col)

    _plot(data, out_path, show=bool(args.show))
    print(f"Wrote slopegraph to {str(out_path)}")


if __name__ == "__main__":
    main()
