from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import clt_raw_comparability_tasklib as tasklib


class _DummyModel:
    def eval(self) -> None:
        return None


def test_build_arg_parser_accepts_revision() -> None:
    parser = tasklib.build_arg_parser(task_default="cf")
    args = parser.parse_args(
        [
            "--model_name_or_path",
            "dummy-model",
            "--clt_repo",
            "dummy-clt",
            "--revision",
            "test-rev",
        ]
    )

    assert args.revision == "test-rev"


def test_run_forwards_revision_to_load_causal_lm(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_load_causal_lm(model_name_or_path: str, device, **kwargs):
        captured["model_name_or_path"] = model_name_or_path
        captured["device"] = str(device)
        captured.update(kwargs)
        return SimpleNamespace(model=_DummyModel(), tokenizer=object())

    monkeypatch.setattr(tasklib, "load_causal_lm", _fake_load_causal_lm)
    monkeypatch.setattr(tasklib, "get_num_layers", lambda model: 13)
    monkeypatch.setattr(tasklib, "_build_cases_for_task", lambda **kwargs: ([], {}, 0))

    parser = tasklib.build_arg_parser(task_default="cf")
    args = parser.parse_args(
        [
            "--model_name_or_path",
            "dummy-model",
            "--clt_repo",
            "dummy-clt",
            "--device",
            "cpu",
            "--revision",
            "test-rev",
        ]
    )

    with pytest.raises(ValueError, match="No protocol cases built"):
        tasklib.run(args)

    assert captured["model_name_or_path"] == "dummy-model"
    assert captured["revision"] == "test-rev"
