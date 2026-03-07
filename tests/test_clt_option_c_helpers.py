from __future__ import annotations

import torch

from aom_clt_feature_analysis import _pooled_std
from aom_clt_head_attribution import _make_head_zero_hooks, _sample_random_heads


def test_pooled_std_matches_manual_two_groups() -> None:
    a = [1.0, 3.0, 5.0]
    b = [2.0, 4.0, 6.0]
    # Both groups have sample variance 4.0 -> pooled std = 2.0
    out = _pooled_std(a, b)
    assert abs(float(out) - 2.0) < 1e-9


def test_make_head_zero_hooks_target_mode_only_zeros_selected_positions() -> None:
    hooks = _make_head_zero_hooks(
        heads_by_layer={2: [1]},
        target_positions=[0, 2],
        ablate_positions="target",
    )
    assert len(hooks) == 1
    act = torch.ones((1, 4, 3, 2), dtype=torch.float32)
    out = hooks[0].fn(act, None)
    assert torch.all(out[:, [0, 2], 1, :] == 0.0)
    assert torch.all(out[:, [1, 3], 1, :] == 1.0)
    assert torch.all(out[:, :, 0, :] == 1.0)
    assert torch.all(out[:, :, 2, :] == 1.0)


def test_make_head_zero_hooks_all_mode_zeros_entire_head() -> None:
    hooks = _make_head_zero_hooks(
        heads_by_layer={1: [0, 2]},
        target_positions=[1],
        ablate_positions="all",
    )
    act = torch.ones((1, 3, 4, 2), dtype=torch.float32)
    out = hooks[0].fn(act, None)
    assert torch.all(out[:, :, 0, :] == 0.0)
    assert torch.all(out[:, :, 2, :] == 0.0)
    assert torch.all(out[:, :, 1, :] == 1.0)
    assert torch.all(out[:, :, 3, :] == 1.0)


def test_sample_random_heads_is_deterministic() -> None:
    all_heads = [(0, h) for h in range(8)]
    excluded = [(0, 0), (0, 1)]
    a = _sample_random_heads(all_heads=all_heads, excluded=excluded, k=3, seed=11)
    b = _sample_random_heads(all_heads=all_heads, excluded=excluded, k=3, seed=11)
    assert a == b
    assert len(a) == 3
    assert all(x not in excluded for x in a)
