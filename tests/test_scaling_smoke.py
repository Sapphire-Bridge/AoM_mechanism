from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

from aom.scaling.layer_map import map_layers
from scripts import run_scaling_study


def _write_yaml(path: Path, obj: dict) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def _csv_header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        return next(r)


def test_layer_map_relative_depth_smoke() -> None:
    lm = map_layers(n_layers=12, strategy="relative_depth", positions=[0.25, 0.5, 0.75, 0.9])
    assert tuple(lm.layers) == (3, 6, 8, 10)


def test_run_scaling_study_smoke_dry_run(tmp_path: Path, monkeypatch) -> None:
    cfg = {
        "scaling": {
            "study_name": "unit_scaling",
            "models": [
                {
                    "label": "tiny-a",
                    "model_name_or_path": "gpt2",
                    "backend": "transformer_lens",
                    "device": "cpu",
                    "n_layers": 4,
                    "local_files_only": True,
                },
                {
                    "label": "tiny-b",
                    "model_name_or_path": "gpt2",
                    "backend": "transformer_lens",
                    "device": "cpu",
                    "n_layers": 6,
                    "local_files_only": True,
                },
            ],
            "tasks": ["disamb", "cf", "coh"],
            "bootstrap_n": 20,
            "bootstrap_seed": 0,
            "ci": 0.95,
            "data": {
                "disamb_path": "data/disamb_pairs.jsonl",
                "cf_path": "data/counterfactual.jsonl",
                "coh_path": "data/coherence.jsonl",
            },
            "layer_map": {"strategy": "relative_depth", "positions": [0.25, 0.5, 0.75]},
            "battery": {
                "eval": True,
                "patching": True,
                "sae_sterility": False,
                "feature_families": False,
                "disamb_path_decomp": True,
                "why_fetch": True,
                "completeness": True,
            },
            "why_fetch": {"n_examples": 5, "heads_topk": 2},
            "disamb_path_decomp": {"max_pairs": 5, "mode": "layer", "position": -1},
            "completeness": {"explanation_path": ""},
        }
    }

    cfg_path = tmp_path / "scaling.yaml"
    _write_yaml(cfg_path, cfg)

    results_root = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scaling_study.py",
            "--config",
            str(cfg_path),
            "--results_root",
            str(results_root),
            "--dry_run",
            "--smoke",
        ],
    )
    run_scaling_study.main()

    study_root = results_root / "unit_scaling"
    assert study_root.exists()

    study_manifest_candidates = sorted(study_root.glob("*/RUN_MANIFEST.json"))
    assert len(study_manifest_candidates) == 1
    study_manifest_path = study_manifest_candidates[0]
    study_manifest = json.loads(study_manifest_path.read_text(encoding="utf-8"))

    outputs = study_manifest.get("outputs", {})
    metrics_path = Path(str(outputs.get("metrics", "")))
    mechanistic_path = Path(str(outputs.get("mechanistic", "")))
    completeness_path = Path(str(outputs.get("completeness", "")))

    for p in (metrics_path, mechanistic_path, completeness_path):
        assert p.exists()
        header = _csv_header(p)
        assert "study_name" in header
        assert "model_label" in header
        assert "model_name_or_path" in header
        assert "run_dir" in header
        assert "source_file" in header

    for label in ("tiny-a", "tiny-b"):
        per_model = sorted((study_root / label).glob("*/RUN_MANIFEST.json"))
        assert len(per_model) == 1
        m = json.loads(per_model[0].read_text(encoding="utf-8"))
        assert m.get("run_status") == "PASS"
        assert int(m.get("run_summary", {}).get("attempted", 0)) > 0
