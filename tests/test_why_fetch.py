from __future__ import annotations

import torch

from aom.mechanistic.why_fetch import aggregate_top_heads, compute_fetch_mass, make_uniform_pattern_hook


def test_compute_fetch_mass_and_aggregate():
    patt = torch.zeros(1, 2, 4, 6, dtype=torch.float32)
    # Head 0 attends fully to token 2 at q=1, head 1 attends fully to token 4.
    patt[0, 0, 1, 2] = 1.0
    patt[0, 1, 1, 4] = 1.0
    cache = {"pattern.0": patt}
    mass = compute_fetch_mass(cache, q_pos=1, evidence_span=(2, 5))
    assert 0 in mass
    assert mass[0].shape == (2,)
    assert torch.allclose(mass[0], torch.tensor([1.0, 1.0]))

    rows = [
        {"task": "disamb", "layer": 0, "head": 0, "fetch_mass": 0.4, "delta_qk": 0.2, "delta_ov": 0.1},
        {"task": "disamb", "layer": 0, "head": 0, "fetch_mass": 0.6, "delta_qk": 0.0, "delta_ov": 0.3},
        {"task": "disamb", "layer": 0, "head": 1, "fetch_mass": 0.2, "delta_qk": 0.1, "delta_ov": 0.05},
    ]
    summary = aggregate_top_heads(rows, bootstrap_n=50, bootstrap_seed=0, ci=0.95)
    assert summary
    assert summary[0]["fetch_mass_mean"] >= summary[-1]["fetch_mass_mean"]
    assert {"delta_qk_mean", "delta_ov_mean", "fetch_mass_mean"} <= set(summary[0].keys())


def test_uniform_pattern_hook_changes_fetch_mass():
    patt = torch.zeros(1, 2, 4, 8, dtype=torch.float32)
    patt[:, 0, :, 1] = 1.0
    patt[:, 1, :, 6] = 1.0
    before = compute_fetch_mass({"pattern.0": patt}, q_pos=0, evidence_span=(0, 4))[0]

    hook = make_uniform_pattern_hook(layer=0, head=0)
    after_patt = hook.fn(patt, None)
    after = compute_fetch_mass({"pattern.0": after_patt}, q_pos=0, evidence_span=(0, 4))[0]

    # Head 0 should change under uniformization; head 1 should remain unchanged.
    assert float(before[0].item()) != float(after[0].item())
    assert float(before[1].item()) == float(after[1].item())
