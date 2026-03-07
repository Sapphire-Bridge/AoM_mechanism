from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..data.schemas import CounterfactualPair
from ..utils import bootstrap_ci_metric
from .disamb import score_labels_next_continuations


@torch.no_grad()
def compute_aom_cf(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: List[CounterfactualPair],
    device: torch.device,
    *,
    normalize_by_length: bool = True,
    ci: float = 0.95,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    """
    Minimal-pair sensitivity: does a targeted context intervention shift label preference?
    """
    def _boot(name: str, values: List[float]) -> Dict[str, Any]:
        mv, lo, hi = bootstrap_ci_metric(values, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed)
        out: Dict[str, Any] = {
            name: float(mv.value),
            f"{name}_ci_low": float(lo),
            f"{name}_ci_high": float(hi),
            f"{name}_n": int(mv.n),
            f"{name}_valid": bool(mv.valid),
        }
        if mv.reason is not None:
            out[f"{name}_reason"] = str(mv.reason)
        return out

    label_acc_all: List[float] = []
    label_acc_shift: List[float] = []
    label_acc_invariant: List[float] = []
    label_acc_graded: List[float] = []

    shift_l1_all: List[float] = []
    shift_l1_shift: List[float] = []
    shift_l1_invariant: List[float] = []
    shift_l1_graded: List[float] = []

    abs_shift_pref_shift: List[float] = []
    abs_shift_pref_invariant: List[float] = []
    abs_shift_pref_graded: List[float] = []
    direction_samples_shift: List[float] = []
    direction_samples_graded: List[float] = []

    for it in items:
        base_scores = score_labels_next_continuations(
            model, tokenizer, it.base.prompt, it.choices, device, normalize_by_length=normalize_by_length
        )
        cf_scores = score_labels_next_continuations(
            model, tokenizer, it.cf.prompt, it.choices, device, normalize_by_length=normalize_by_length
        )

        base_pred = base_scores.argmax_label()
        cf_pred = cf_scores.argmax_label()
        base_correct = float(base_pred == it.base.expected_label)
        cf_correct = float(cf_pred == it.cf.expected_label)
        # Bootstrap unit is the minimal-pair item (not the prompt side).
        item_label_acc = 0.5 * (base_correct + cf_correct)
        label_acc_all.append(item_label_acc)
        if it.expected_effect == "shift":
            label_acc_shift.append(item_label_acc)
        elif it.expected_effect == "invariant":
            label_acc_invariant.append(item_label_acc)
        elif it.expected_effect == "graded":
            label_acc_graded.append(item_label_acc)

        # A simple shift magnitude proxy: L1 distance over label-score vectors.
        labels = sorted(set(base_scores.by_label) | set(cf_scores.by_label))
        shift = sum(abs(base_scores.by_label.get(l, 0.0) - cf_scores.by_label.get(l, 0.0)) for l in labels)
        shift = float(shift)
        shift_l1_all.append(shift)
        if it.expected_effect == "shift":
            shift_l1_shift.append(shift)
        elif it.expected_effect == "invariant":
            shift_l1_invariant.append(shift)
        elif it.expected_effect == "graded":
            shift_l1_graded.append(shift)

        # Preference-shift metric:
        # shift_pref = [score_cf(L1) - score_cf(L0)] - [score_base(L1) - score_base(L0)]
        contrast = it.contrast_labels
        if contrast is None and it.base.expected_label != it.cf.expected_label:
            contrast = (it.base.expected_label, it.cf.expected_label)
        if contrast is not None:
            L0, L1 = contrast
            base_pref = base_scores.by_label.get(L1, 0.0) - base_scores.by_label.get(L0, 0.0)
            cf_pref = cf_scores.by_label.get(L1, 0.0) - cf_scores.by_label.get(L0, 0.0)
            shift_pref = float(cf_pref - base_pref)
            abs_shift_pref = abs(shift_pref)

            if it.expected_effect == "shift":
                abs_shift_pref_shift.append(abs_shift_pref)
                direction_samples_shift.append(float(shift_pref > 0.0))
            elif it.expected_effect == "invariant":
                abs_shift_pref_invariant.append(abs_shift_pref)
            elif it.expected_effect == "graded":
                abs_shift_pref_graded.append(abs_shift_pref)
                direction_samples_graded.append(float(shift_pref > 0.0))

    n_items_total = len(items)
    n_items_shift = sum(1 for it in items if it.expected_effect == "shift")
    n_items_invariant = sum(1 for it in items if it.expected_effect == "invariant")
    n_items_graded = sum(1 for it in items if it.expected_effect == "graded")

    out: Dict[str, Any] = {
        # Secondary: label accuracy on base+cf prompts (per-item averaged).
        **_boot("label_accuracy", label_acc_all),
        **_boot("label_accuracy_shift_items", label_acc_shift),
        **_boot("label_accuracy_invariant_items", label_acc_invariant),
        **_boot("label_accuracy_graded_items", label_acc_graded),
        # Primary AoM-CF (paper): whether the intervention shifts preference in the expected direction.
        **_boot("shift_direction_accuracy", direction_samples_shift),
        **_boot("graded_direction_accuracy", direction_samples_graded),
        **_boot("mean_shift_l1", shift_l1_all),
        **_boot("mean_shift_l1_shift_items", shift_l1_shift),
        **_boot("mean_shift_l1_invariant_items", shift_l1_invariant),
        **_boot("mean_shift_l1_graded_items", shift_l1_graded),
        # Mean absolute preference shift, split by expected effect.
        **_boot("mean_abs_shift_pref_shift_items", abs_shift_pref_shift),
        **_boot("mean_abs_shift_pref_invariant_items", abs_shift_pref_invariant),
        **_boot("mean_abs_shift_pref_graded_items", abs_shift_pref_graded),
        "n_items_total": int(n_items_total),
        "n_items_shift": int(n_items_shift),
        "n_items_invariant": int(n_items_invariant),
        "n_items_graded": int(n_items_graded),
    }
    return out
