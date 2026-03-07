from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mom_circuits_tlens import _load_manifest, _load_tlens, _sample_random_heads


def test_load_tlens_dependency_guard():
    if importlib.util.find_spec("transformer_lens") is not None:
        pytest.skip("transformer_lens installed; dependency guard not applicable")
    with pytest.raises(RuntimeError):
        _load_tlens()


def test_sample_random_heads_is_deterministic_and_excludes():
    h1 = _sample_random_heads(n_heads=8, k=3, excluded=[0, 1], seed=123)
    h2 = _sample_random_heads(n_heads=8, k=3, excluded=[0, 1], seed=123)
    assert h1 == h2
    assert len(h1) == 3
    assert all(0 <= x < 8 for x in h1)
    assert all(x not in {0, 1} for x in h1)


def test_load_manifest_accepts_object_and_list(tmp_path: Path):
    obj_path = tmp_path / "m_obj.json"
    lst_path = tmp_path / "m_list.json"

    obj_path.write_text(
        json.dumps(
            {
                "manifest_version": "1.0",
                "items": [
                    {
                        "pair_id": "p0",
                        "prompt_text": "x",
                        "target_pos": 0,
                        "refusal_token_id": 1,
                        "guidance_token_id": 2,
                        "selected_layers": [0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    lst_path.write_text(
        json.dumps(
            [
                {
                    "pair_id": "p1",
                    "prompt_text": "x",
                    "target_pos": 0,
                    "refusal_token_id": 1,
                    "guidance_token_id": 2,
                    "selected_layers": [0],
                }
            ]
        ),
        encoding="utf-8",
    )

    meta_obj, items_obj = _load_manifest(obj_path)
    meta_lst, items_lst = _load_manifest(lst_path)

    assert meta_obj["manifest_version"] == "1.0"
    assert meta_lst["manifest_version"] == "legacy-list"
    assert items_obj[0]["pair_id"] == "p0"
    assert items_lst[0]["pair_id"] == "p1"
