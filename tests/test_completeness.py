from __future__ import annotations

from aom.metrics import completeness as comp


def test_completeness_total_equals_explained(monkeypatch):
    monkeypatch.setattr(comp, "get_num_layers", lambda _m: 4)
    monkeypatch.setattr(comp, "_protocol_for_task", lambda _t: object())

    def _fake_run_activation_patching(**kwargs):
        layers = list(kwargs.get("layers", []))
        if layers == [0, 1, 2]:
            return {"mean_max_effect": 1.0, "mean_max_effect_ci_low": 0.8, "mean_max_effect_ci_high": 1.2}
        if layers == [0, 1, 2]:
            return {"mean_max_effect": 1.0, "mean_max_effect_ci_low": 0.8, "mean_max_effect_ci_high": 1.2}
        return {"mean_max_effect": 1.0, "mean_max_effect_ci_low": 0.8, "mean_max_effect_ci_high": 1.2}

    monkeypatch.setattr(comp, "run_activation_patching", _fake_run_activation_patching)

    res = comp.compute_task_completeness(
        model=object(),
        tokenizer=object(),
        items=[],
        task="disamb",
        device=None,  # type: ignore[arg-type]
        total_layers=[0, 1, 2],
        explanation={"disamb": {"heads": [{"layer": 0}, {"layer": 1}, {"layer": 2}]}},
    )
    assert abs(float(res.completeness) - 1.0) < 1e-8
    assert abs(float(res.dark_matter)) < 1e-8


def test_completeness_empty_explanation(monkeypatch):
    monkeypatch.setattr(comp, "get_num_layers", lambda _m: 4)
    monkeypatch.setattr(comp, "_protocol_for_task", lambda _t: object())

    def _fake_run_activation_patching(**kwargs):
        _ = kwargs
        return {"mean_max_effect": 2.0, "mean_max_effect_ci_low": 1.5, "mean_max_effect_ci_high": 2.5}

    monkeypatch.setattr(comp, "run_activation_patching", _fake_run_activation_patching)

    res = comp.compute_task_completeness(
        model=object(),
        tokenizer=object(),
        items=[],
        task="cf",
        device=None,  # type: ignore[arg-type]
        total_layers=[0, 1, 2],
        explanation={},
    )
    assert abs(float(res.completeness)) < 1e-8
    assert abs(float(res.dark_matter) - 2.0) < 1e-8
