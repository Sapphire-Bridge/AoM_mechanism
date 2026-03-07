from __future__ import annotations

import json
from pathlib import Path

import pytest

from aom.data.bundle_manifest import compute_bundle_id, validate_bundle_manifest
from aom.data.dataset_manifest import sha256_file


def test_compute_bundle_id_stable_across_metadata() -> None:
    files = {
        "disamb_pairs.jsonl": {"sha256": "a" * 64},
        "counterfactual.jsonl": {"sha256": "b" * 64},
        "coherence.jsonl": {"sha256": "c" * 64},
    }
    m1 = {"name": "paper_hardened_v2", "generated_at_utc": "t1", "files": files}
    m2 = {"name": "paper_hardened_v2", "generated_at_utc": "t2", "git_commit": "x", "files": files}
    assert compute_bundle_id(m1) == compute_bundle_id(m2)


def test_compute_bundle_id_changes_when_any_file_hash_changes() -> None:
    files1 = {
        "disamb_pairs.jsonl": {"sha256": "a" * 64},
        "counterfactual.jsonl": {"sha256": "b" * 64},
        "coherence.jsonl": {"sha256": "c" * 64},
    }
    files2 = dict(files1)
    files2["counterfactual.jsonl"] = {"sha256": "d" * 64}
    assert compute_bundle_id({"files": files1}) != compute_bundle_id({"files": files2})


def test_compute_bundle_id_depends_on_file_identity_not_multiset() -> None:
    m1 = {
        "files": {
            "disamb_pairs.jsonl": {"sha256": "a" * 64},
            "counterfactual.jsonl": {"sha256": "b" * 64},
            "coherence.jsonl": {"sha256": "c" * 64},
        }
    }
    m2 = {
        "files": {
            "disamb_pairs.jsonl": {"sha256": "b" * 64},
            "counterfactual.jsonl": {"sha256": "a" * 64},
            "coherence.jsonl": {"sha256": "c" * 64},
        }
    }
    assert compute_bundle_id(m1) != compute_bundle_id(m2)


def test_validate_bundle_manifest_returns_bundle_id_and_checks_hashes(tmp_path: Path) -> None:
    dis = tmp_path / "disamb_pairs.jsonl"
    cf = tmp_path / "counterfactual.jsonl"
    coh = tmp_path / "coherence.jsonl"
    dis.write_text("x\n", encoding="utf-8")
    cf.write_text("y\n", encoding="utf-8")
    coh.write_text("z\n", encoding="utf-8")

    manifest = {
        "name": "paper_hardened_v2",
        "files": {
            "disamb_pairs.jsonl": {"sha256": sha256_file(dis)},
            "counterfactual.jsonl": {"sha256": sha256_file(cf)},
            "coherence.jsonl": {"sha256": sha256_file(coh)},
        },
    }
    mp = tmp_path / "DATASET_MANIFEST.json"
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out = validate_bundle_manifest(mp, disamb_path=str(dis), cf_path=str(cf), coh_path=str(coh))
    assert out["dataset_bundle_manifest_name"] == "paper_hardened_v2"
    assert out["dataset_bundle_id"] == compute_bundle_id(manifest)

    cf.write_text("y \n", encoding="utf-8")  # 1-byte change (trailing space)
    with pytest.raises(ValueError) as ei:
        validate_bundle_manifest(mp, disamb_path=str(dis), cf_path=str(cf), coh_path=str(coh))
    msg = str(ei.value)
    assert "role=cf" in msg
    assert str(cf) in msg


def test_validate_bundle_manifest_rejects_mismatched_bundle_id(tmp_path: Path) -> None:
    dis = tmp_path / "disamb_pairs.jsonl"
    dis.write_text("x\n", encoding="utf-8")

    manifest = {
        "name": "paper_hardened_v2",
        "bundle_id": "0" * 64,
        "files": {"disamb_pairs.jsonl": {"sha256": sha256_file(dis)}},
    }
    mp = tmp_path / "DATASET_MANIFEST.json"
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_bundle_manifest(mp, disamb_path=str(dis))
