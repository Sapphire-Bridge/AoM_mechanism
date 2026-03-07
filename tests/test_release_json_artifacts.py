from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

from scripts import release_json_artifacts as rja


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    return repo


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_sanitize_and_audit_json_changes_against_explicit_baseline(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    artifact_path = repo / "artifact.json"

    baseline = {
        "csv_path": f"{repo}/results/out.csv",
        "model_name_or_path": (
            "/cache/huggingface/hub/models--google--gemma-2-2b/"
            "snapshots/c5ebcd40d208330abc697524c919956e692655cf"
        ),
        "argv_redacted": ["python", f"{repo}/scripts/run.py"],
        "argv_redacted_json": json.dumps(["python", f"{repo}/scripts/run.py"]),
        "argv_sha256": "stale",
        "score_mean": 1.25,
    }
    artifact_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    baseline_ref = _commit_all(repo, "baseline json")

    assert (
        rja.main(
            [
                "sanitize",
                "--repo_root",
                str(repo),
                "--baseline_ref",
                baseline_ref,
                "artifact.json",
            ]
        )
        == 0
    )
    assert (
        rja.main(
            [
                "audit",
                "--repo_root",
                str(repo),
                "--baseline_ref",
                baseline_ref,
                "--changed_only",
            ]
        )
        == 0
    )

    sanitized = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert sanitized["csv_path"] == "results/out.csv"
    assert sanitized["model_name_or_path"] == "hf://google/gemma-2-2b@c5ebcd40d208330abc697524c919956e692655cf"
    assert sanitized["argv_redacted_json"] == json.dumps(["python", "scripts/run.py"])

    sanitized["score_mean"] = 2.0
    artifact_path.write_text(json.dumps(sanitized, indent=2) + "\n", encoding="utf-8")
    assert (
        rja.main(
            [
                "audit",
                "--repo_root",
                str(repo),
                "--baseline_ref",
                baseline_ref,
                "--changed_only",
            ]
        )
        == 2
    )


def test_sanitize_csv_row_refreshes_hashes_and_repo_relative_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    manifest_path = repo_root / "data" / "DATASET_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"name":"demo"}\n', encoding="utf-8")

    row = {
        "csv_path": f"{repo_root}/results/out.csv",
        "dataset_manifest_path": f"{repo_root}/data/DATASET_MANIFEST.json",
        "argv_redacted_json": json.dumps([f"{repo_root}/scripts/run.py"]),
        "argv_sha256": "stale",
        "dataset_bundle_manifest_sha256": "stale",
    }
    sanitized = rja._sanitize_csv_row(row, repo_root)

    expected_argv_json = json.dumps(["scripts/run.py"])
    assert sanitized["csv_path"] == "results/out.csv"
    assert sanitized["dataset_manifest_path"] == "data/DATASET_MANIFEST.json"
    assert sanitized["argv_redacted_json"] == expected_argv_json
    assert sanitized["argv_sha256"] == hashlib.sha256(expected_argv_json.encode("utf-8")).hexdigest()
    assert sanitized["dataset_bundle_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


def test_sanitize_csv_row_rewrites_historical_repo_absolute_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    historical_repo_root = "/Users/felixb/aom_prototype"
    manifest_path = repo_root / "data" / "DATASET_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"name":"demo"}\n', encoding="utf-8")

    row = {
        "dataset_manifest_path": f"{historical_repo_root}/data/DATASET_MANIFEST.json",
        "protocol_path": f"{historical_repo_root}/configs/protocol.yaml",
    }

    sanitized = rja._sanitize_csv_row(row, repo_root)

    assert sanitized["dataset_manifest_path"] == "data/DATASET_MANIFEST.json"
    assert sanitized["protocol_path"] == "configs/protocol.yaml"


def test_audit_csv_allows_path_and_provenance_changes_but_rejects_metric_drift(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    artifact_path = repo / "artifact.csv"

    with artifact_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "csv_path",
                "model_name_or_path",
                "argv_redacted_json",
                "argv_sha256",
                "score_mean",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "csv_path": f"{repo}/results/out.csv",
                "model_name_or_path": (
                    "/cache/huggingface/hub/models--google--gemma-2-2b/"
                    "snapshots/c5ebcd40d208330abc697524c919956e692655cf"
                ),
                "argv_redacted_json": json.dumps([f"{repo}/scripts/run.py"]),
                "argv_sha256": "stale",
                "score_mean": "1.25",
            }
        )
    baseline_ref = _commit_all(repo, "baseline csv")

    assert (
        rja.main(
            [
                "sanitize-csv",
                "--repo_root",
                str(repo),
                "--baseline_ref",
                baseline_ref,
                "artifact.csv",
            ]
        )
        == 0
    )
    assert (
        rja.main(
            [
                "audit-csv",
                "--repo_root",
                str(repo),
                "--baseline_ref",
                baseline_ref,
                "--changed_only",
            ]
        )
        == 0
    )

    rows = list(csv.DictReader(artifact_path.read_text(encoding="utf-8").splitlines()))
    rows[0]["score_mean"] = "9.99"
    with artifact_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    assert (
        rja.main(
            [
                "audit-csv",
                "--repo_root",
                str(repo),
                "--baseline_ref",
                baseline_ref,
                "--changed_only",
            ]
        )
        == 2
    )


def test_audit_changed_only_skips_files_missing_from_baseline(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
    baseline_ref = _commit_all(repo, "baseline")

    (repo / "new_artifact.json").write_text('{"value": 1}\n', encoding="utf-8")
    _git(repo, "add", "new_artifact.json")

    assert (
        rja.main(
            [
                "audit",
                "--repo_root",
                str(repo),
                "--baseline_ref",
                baseline_ref,
                "--changed_only",
            ]
        )
        == 0
    )


def test_publish_writes_non_mutating_publication_copies_and_audits(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    artifact_json = repo / "artifact.json"
    artifact_csv = repo / "artifact.csv"

    artifact_json.write_text(
        json.dumps(
            {
                "csv_path": f"{repo}/results/out.csv",
                "model_name_or_path": (
                    "/cache/huggingface/hub/models--google--gemma-2-2b/"
                    "snapshots/c5ebcd40d208330abc697524c919956e692655cf"
                ),
                "score_mean": math.nan,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with artifact_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["csv_path", "model_name_or_path", "score_mean", "composite_missing_policy"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "csv_path": f"{repo}/results/out.csv",
                "model_name_or_path": (
                    "/cache/huggingface/hub/models--google--gemma-2-2b/"
                    "snapshots/c5ebcd40d208330abc697524c919956e692655cf"
                ),
                "score_mean": "nan",
                "composite_missing_policy": "nan",
            }
        )

    source_json_bytes = artifact_json.read_bytes()
    source_csv_bytes = artifact_csv.read_bytes()

    out_dir = repo / "release_public"
    assert (
        rja.main(
            [
                "publish",
                "--repo_root",
                str(repo),
                "--out_dir",
                str(out_dir),
                "artifact.json",
                "artifact.csv",
            ]
        )
        == 0
    )

    assert artifact_json.read_bytes() == source_json_bytes
    assert artifact_csv.read_bytes() == source_csv_bytes

    published_json = json.loads((out_dir / "artifact.json").read_text(encoding="utf-8"))
    published_csv_text = (out_dir / "artifact.csv").read_text(encoding="utf-8")

    assert published_json["csv_path"] == "results/out.csv"
    assert published_json["model_name_or_path"] == "hf://google/gemma-2-2b@c5ebcd40d208330abc697524c919956e692655cf"
    assert published_json["score_mean"] is None
    assert "NaN" not in (out_dir / "artifact.json").read_text(encoding="utf-8")

    assert "results/out.csv" in published_csv_text
    assert "hf://google/gemma-2-2b@c5ebcd40d208330abc697524c919956e692655cf" in published_csv_text
    assert ",null,nan\n" in published_csv_text

    json_audit = json.loads((out_dir / "audit" / "json_audit.json").read_text(encoding="utf-8"))
    csv_audit = json.loads((out_dir / "audit" / "csv_audit.json").read_text(encoding="utf-8"))
    portability = json.loads((out_dir / "audit" / "portability_scan.json").read_text(encoding="utf-8"))

    assert json_audit["status"] == "ok"
    assert csv_audit["status"] == "ok"
    assert portability["status"] == "ok"
    assert (out_dir / "RELEASE_MANIFEST.json").exists()
    assert rja._scan_portability(out_dir) == []
    for path in sorted(out_dir.rglob("*.json")):
        text = path.read_text(encoding="utf-8")
        assert "NaN" not in text
        assert "Infinity" not in text
        assert "/Users/" not in text
        assert "/home/" not in text


def test_publish_is_deterministic(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "artifact.json").write_text(
        json.dumps({"csv_path": f"{repo}/results/out.csv", "score_mean": math.nan}, indent=2) + "\n",
        encoding="utf-8",
    )

    out_one = repo / "release_public_one"
    out_two = repo / "release_public_two"
    for out_dir in (out_one, out_two):
        assert (
            rja.main(
                [
                    "publish",
                    "--repo_root",
                    str(repo),
                    "--out_dir",
                    str(out_dir),
                    "artifact.json",
                ]
            )
            == 0
        )

    files_one = sorted(path.relative_to(out_one).as_posix() for path in out_one.rglob("*") if path.is_file())
    files_two = sorted(path.relative_to(out_two).as_posix() for path in out_two.rglob("*") if path.is_file())
    assert files_one == files_two
    for rel_path in files_one:
        assert (out_one / rel_path).read_bytes() == (out_two / rel_path).read_bytes()


def test_change_classifier_distinguishes_portability_path_provenance_and_metric() -> None:
    assert rja._classify_change("score_mean", math.nan, None) == "portability-only"
    assert rja._classify_change("csv_path", "/tmp/repo/results/out.csv", "results/out.csv") == "path-only"
    assert (
        rja._classify_change(
            "model_name_or_path",
            "/cache/huggingface/hub/models--google--gemma-2-2b/snapshots/rev",
            "hf://google/gemma-2-2b@rev",
        )
        == "provenance-semantic"
    )
    assert rja._classify_change("score_mean", 1.25, 2.0) == "metric-semantic"
