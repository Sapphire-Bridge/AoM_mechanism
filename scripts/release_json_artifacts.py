from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


HASH_BY_DEPENDENCY_KEYS = {
    "argv_sha256",
    "dataset_bundle_manifest_sha256",
    "comparability_summary_sha256",
}
PROVENANCE_SEMANTIC_KEYS = {
    "model_name_or_path",
    "model",
    "clt_repo",
    "sae_repo",
    "argv_redacted",
    "argv_redacted_json",
}
PATH_ONLY_SUFFIXES = (
    "path",
    "repo_root",
)
NONFINITE_CSV_SENTINEL = "null"
NONFINITE_TEXT_ALLOWLIST_KEYS = {
    "composite_missing_policy",
}
PORTABILITY_SCAN_PATTERNS = {
    "bare_nan": re.compile(r"(?<![A-Za-z0-9_])NaN(?![A-Za-z0-9_])"),
    "bare_pos_infinity": re.compile(r"(?<![A-Za-z0-9_-])Infinity(?![A-Za-z0-9_])"),
    "bare_neg_infinity": re.compile(r"(?<![A-Za-z0-9_])-Infinity(?![A-Za-z0-9_])"),
    "users_path": re.compile(r"/Users/"),
    "home_path": re.compile(r"/home/"),
    "windows_abs_path": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
}
APPENDIX_A2_MARKER = "### A.2 Claim-to-artifact map"
BACKTICK_RE = re.compile(r"`([^`]+)`")
TABLE_SEPARATOR_RE = re.compile(r"^\|\s*-{3,}\s*(\|\s*-{3,}\s*)+\|?$")
HF_SNAPSHOT_RE = re.compile(
    r"^(?P<prefix>.*?[/\\]huggingface[/\\]hub[/\\]models--"
    r"(?P<org>[^/\\]+)--(?P<name>[^/\\]+)[/\\]snapshots[/\\](?P<rev>[^/\\]+))"
    r"(?P<suffix>(?:[/\\].*)?)$"
)
REPO_ROOT_MARKERS = {
    "AoM_interpretability",
    "aom_prototype",
}


@dataclass(frozen=True)
class JsonChange:
    path: str
    category: str
    old_value: Any
    new_value: Any


@dataclass(frozen=True)
class PublicationPath:
    rel_path: str
    kind: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run_git(repo_root: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _tracked_paths(repo_root: Path, baseline_ref: str, glob: str) -> list[Path]:
    out = _run_git(repo_root, ["diff", "--name-only", baseline_ref, "--", glob])
    return [repo_root / line.strip() for line in out.splitlines() if line.strip()]


def _tracked_json_paths(repo_root: Path, baseline_ref: str) -> list[Path]:
    return _tracked_paths(repo_root, baseline_ref, "*.json")


def _tracked_csv_paths(repo_root: Path, baseline_ref: str) -> list[Path]:
    return _tracked_paths(repo_root, baseline_ref, "*.csv")


def _load_json_from_ref(repo_root: Path, rel_path: Path, baseline_ref: str) -> Any:
    blob = _run_git(repo_root, ["show", f"{baseline_ref}:{rel_path.as_posix()}"])
    return json.loads(blob)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _portable_json_value(data: Any) -> Any:
    if _is_nonfinite_number(data):
        return None
    if isinstance(data, dict):
        return {str(key): _portable_json_value(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_portable_json_value(item) for item in data]
    return data


def _dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_portable_json_value(data), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _path_exists_in_ref(repo_root: Path, rel_path: Path, ref: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{ref}:{rel_path.as_posix()}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _extract_appendix_a2_section(paper_text: str) -> str:
    start = paper_text.find(APPENDIX_A2_MARKER)
    if start == -1:
        return ""
    tail = paper_text[start:]
    next_heading = re.search(r"^###\s", tail[len(APPENDIX_A2_MARKER) :], flags=re.MULTILINE)
    if next_heading is None:
        return tail
    return tail[: len(APPENDIX_A2_MARKER) + next_heading.start()]


def _default_publication_paths(repo_root: Path, paper_path: Path) -> list[PublicationPath]:
    paper_text = paper_path.read_text(encoding="utf-8")
    appendix = _extract_appendix_a2_section(paper_text)
    rels: set[tuple[str, str]] = set()
    for line in appendix.splitlines():
        if not line.startswith("|"):
            continue
        if TABLE_SEPARATOR_RE.fullmatch(line.strip()) is not None:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] == "Evidence ID(s)" or len(cells) < 3:
            continue
        for token in BACKTICK_RE.findall(cells[2]):
            if token.endswith(".json"):
                rels.add((token, "json"))
            elif token.endswith(".csv"):
                rels.add((token, "csv"))

    data_manifest = repo_root / "data_paper_hardened_v2" / "DATASET_MANIFEST.json"
    if data_manifest.exists():
        rels.add(("data_paper_hardened_v2/DATASET_MANIFEST.json", "json"))

    return [PublicationPath(rel_path=rel_path, kind=kind) for rel_path, kind in sorted(rels)]


def _scan_portability(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PORTABILITY_SCAN_PATTERNS.items():
            if pattern.search(text) is None:
                continue
            findings.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "pattern": label,
                }
            )
    return findings


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_nonfinite_number(value: Any) -> bool:
    return isinstance(value, float) and not math.isfinite(value)


def _is_nonfinite_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return normalized in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}


def _is_portability_normalization(old_value: Any, new_value: Any) -> bool:
    if _is_nonfinite_number(old_value) and new_value is None:
        return True
    if _is_nonfinite_text(old_value) and new_value == NONFINITE_CSV_SENTINEL:
        return True
    return False


def _looks_like_repo_relative(path_value: str) -> bool:
    if not path_value:
        return False
    if path_value.startswith("./") or path_value.startswith("../"):
        return True
    return not Path(path_value).is_absolute()


def _sanitize_hf_snapshot(path_value: str) -> str | None:
    match = HF_SNAPSHOT_RE.fullmatch(path_value)
    if match is None:
        return None
    org = match.group("org")
    name = match.group("name")
    rev = match.group("rev")
    suffix = str(match.group("suffix") or "").replace("\\", "/")
    return f"hf://{org}/{name}@{rev}{suffix}"


def _sanitize_string(value: str, repo_root: Path) -> str:
    sanitized_hf = _sanitize_hf_snapshot(value)
    if sanitized_hf is not None:
        return sanitized_hf

    repo_root_str = str(repo_root)
    if value == repo_root_str:
        return "."
    if value.startswith(repo_root_str + "/"):
        return value[len(repo_root_str) + 1 :]
    if value.startswith(repo_root_str + "\\"):
        return value[len(repo_root_str) + 1 :].replace("\\", "/")

    historical_repo_relative = _sanitize_historical_repo_path(value, repo_root)
    if historical_repo_relative is not None:
        return historical_repo_relative
    return value


def _sanitize_historical_repo_path(value: str, repo_root: Path) -> str | None:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not path.is_absolute():
        return None

    parts = path.parts
    repo_entries = {child.name for child in repo_root.iterdir()}
    markers = set(REPO_ROOT_MARKERS) | {repo_root.name}

    for idx, part in enumerate(parts):
        if part not in markers:
            continue
        suffix = parts[idx + 1 :]
        if not suffix:
            return "."
        return PurePosixPath(*suffix).as_posix()

    preferred_roots = {
        "aom",
        "clt_bundles",
        "configs",
        "data",
        "data_paper_hardened_v2",
        "paper",
        "results",
        "scripts",
        "tables",
        "CITATION.cff",
        "LICENSE",
        "Makefile",
        "README.md",
    }
    for idx, part in enumerate(parts):
        if part not in repo_entries and part not in preferred_roots:
            continue
        suffix = parts[idx:]
        if not suffix:
            continue
        candidate = Path(*suffix)
        if (repo_root / candidate).exists() or part in preferred_roots:
            return PurePosixPath(*suffix).as_posix()
    return None


def _sanitize_obj(obj: Any, repo_root: Path, *, key_path: str = "") -> Any:
    if _is_nonfinite_number(obj):
        return None
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            child_path = f"{key_path}.{key}" if key_path else str(key)
            if key == "argv_redacted_json" and isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    out[key] = _sanitize_string(value, repo_root)
                else:
                    out[key] = json.dumps(_sanitize_obj(parsed, repo_root, key_path=child_path))
                continue
            out[key] = _sanitize_obj(value, repo_root, key_path=child_path)
        return out
    if isinstance(obj, list):
        return [_sanitize_obj(item, repo_root, key_path=key_path) for item in obj]
    if isinstance(obj, str):
        return _sanitize_string(obj, repo_root)
    return obj


def _recompute_hashes(data: Any, path: Path, artifact_root: Path) -> None:
    if not isinstance(data, dict):
        return

    argv_redacted = data.get("argv_redacted")
    if isinstance(argv_redacted, list):
        argv_json = json.dumps(argv_redacted)
        data["argv_redacted_json"] = argv_json
        if "argv_sha256" in data:
            data["argv_sha256"] = hashlib.sha256(argv_json.encode("utf-8")).hexdigest()

    dataset_manifest_path = data.get("dataset_manifest_path")
    if (
        "dataset_bundle_manifest_sha256" in data
        and isinstance(dataset_manifest_path, str)
        and dataset_manifest_path
        and _looks_like_repo_relative(dataset_manifest_path)
    ):
        manifest_path = artifact_root / dataset_manifest_path
        if manifest_path.exists():
            data["dataset_bundle_manifest_sha256"] = _sha256_file(manifest_path)

    hashes = data.get("hashes")
    if isinstance(hashes, dict) and "comparability_summary_sha256" in hashes:
        suffix = ".endpoint_decomp_summary_v2.json"
        if path.name.endswith(suffix):
            summary_path = path.with_name(path.name.replace(suffix, ".summary.json"))
            if summary_path.exists():
                hashes["comparability_summary_sha256"] = _sha256_file(summary_path)


def _sanitize_file(path: Path, repo_root: Path, source_data: Any, *, artifact_root: Path | None = None) -> None:
    sanitized = _sanitize_obj(source_data, repo_root)
    _recompute_hashes(sanitized, path, artifact_root or repo_root)
    _dump_json(path, sanitized)


def _sanitize_csv_row(
    row: dict[str, str],
    repo_root: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in row.items():
        if value is None:
            out[key] = value
            continue
        if key == "argv_redacted_json":
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                out[key] = _sanitize_string(value, repo_root)
            else:
                out[key] = json.dumps(_sanitize_obj(parsed, repo_root, key_path=key))
            continue
        sanitized_value = _sanitize_string(value, repo_root)
        if key not in NONFINITE_TEXT_ALLOWLIST_KEYS and _is_nonfinite_text(sanitized_value):
            out[key] = NONFINITE_CSV_SENTINEL
        else:
            out[key] = sanitized_value

    argv_json = out.get("argv_redacted_json")
    if isinstance(argv_json, str) and "argv_sha256" in out:
        out["argv_sha256"] = hashlib.sha256(argv_json.encode("utf-8")).hexdigest()

    dataset_manifest_path = out.get("dataset_manifest_path")
    if (
        "dataset_bundle_manifest_sha256" in out
        and isinstance(dataset_manifest_path, str)
        and dataset_manifest_path
        and _looks_like_repo_relative(dataset_manifest_path)
    ):
        manifest_path = (artifact_root or repo_root) / dataset_manifest_path
        if manifest_path.exists():
            out["dataset_bundle_manifest_sha256"] = _sha256_file(manifest_path)
    return out


def _sanitize_csv_text(text: str, repo_root: Path, *, artifact_root: Path | None = None) -> str:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return text
    rows = [_sanitize_csv_row(row, repo_root, artifact_root=artifact_root) for row in reader]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _sanitize_csv_file(path: Path, repo_root: Path, source_text: str, *, artifact_root: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_sanitize_csv_text(source_text, repo_root, artifact_root=artifact_root), encoding="utf-8")


def _parse_csv_text(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = list(reader.fieldnames or [])
    rows = [dict(row) for row in reader]
    return fieldnames, rows


def _flatten_changes(old: Any, new: Any, *, path: str = "") -> list[tuple[str, Any, Any]]:
    if isinstance(old, float) and isinstance(new, float) and math.isnan(old) and math.isnan(new):
        return []
    if isinstance(old, dict) and isinstance(new, dict):
        changes: list[tuple[str, Any, Any]] = []
        for key in sorted(set(old) | set(new)):
            child = f"{path}.{key}" if path else str(key)
            if key not in old:
                changes.append((child, None, new[key]))
            elif key not in new:
                changes.append((child, old[key], None))
            else:
                changes.extend(_flatten_changes(old[key], new[key], path=child))
        return changes
    if isinstance(old, list) and isinstance(new, list):
        if old != new:
            return [(path, old, new)]
        return []
    if old != new:
        return [(path, old, new)]
    return []


def _flatten_csv_changes(old_text: str, new_text: str) -> list[tuple[str, Any, Any]]:
    old_fieldnames, old_rows = _parse_csv_text(old_text)
    new_fieldnames, new_rows = _parse_csv_text(new_text)
    changes: list[tuple[str, Any, Any]] = []
    if old_fieldnames != new_fieldnames:
        changes.append(("__schema__.fieldnames", old_fieldnames, new_fieldnames))
    if len(old_rows) != len(new_rows):
        changes.append(("__rows__.count", len(old_rows), len(new_rows)))

    max_rows = max(len(old_rows), len(new_rows))
    for idx in range(max_rows):
        row_path = f"row[{idx}]"
        if idx >= len(old_rows):
            changes.append((row_path, None, new_rows[idx]))
            continue
        if idx >= len(new_rows):
            changes.append((row_path, old_rows[idx], None))
            continue
        old_row = old_rows[idx]
        new_row = new_rows[idx]
        for key in sorted(set(old_row) | set(new_row)):
            old_value = old_row.get(key)
            new_value = new_row.get(key)
            if old_value != new_value:
                changes.append((f"{row_path}.{key}", old_value, new_value))
    return changes


def _classify_change(path: str, old_value: Any, new_value: Any) -> str:
    if _is_portability_normalization(old_value, new_value):
        return "portability-only"
    leaf = path.split(".")[-1]
    if leaf in HASH_BY_DEPENDENCY_KEYS:
        return "hash-by-dependency"
    if leaf in PROVENANCE_SEMANTIC_KEYS:
        return "provenance-semantic"
    if leaf == "repo_root":
        return "path-only"
    if leaf.endswith(PATH_ONLY_SUFFIXES) or "_path_" in leaf:
        return "path-only"
    if isinstance(old_value, str) and isinstance(new_value, str):
        if ("/" in old_value or "\\" in old_value or old_value.startswith("hf://")) and (
            "/" in new_value or "\\" in new_value or new_value.startswith("hf://")
        ):
            return "path-only"
    return "metric-semantic"


def _audit_json_values(old: Any, new: Any) -> list[JsonChange]:
    changes: list[JsonChange] = []
    for key_path, old_value, new_value in _flatten_changes(old, new):
        changes.append(
            JsonChange(
                path=key_path,
                category=_classify_change(key_path, old_value, new_value),
                old_value=old_value,
                new_value=new_value,
            )
        )
    return changes


def _audit_file(repo_root: Path, path: Path, baseline_ref: str) -> list[JsonChange]:
    old = _load_json_from_ref(repo_root, path.relative_to(repo_root), baseline_ref)
    new = _load_json(path)
    return _audit_json_values(old, new)


def _audit_csv_values(old_text: str, new_text: str) -> list[JsonChange]:
    changes: list[JsonChange] = []
    for key_path, old_value, new_value in _flatten_csv_changes(old_text, new_text):
        changes.append(
            JsonChange(
                path=key_path,
                category=_classify_change(key_path, old_value, new_value),
                old_value=old_value,
                new_value=new_value,
            )
        )
    return changes


def _audit_csv_file(repo_root: Path, path: Path, baseline_ref: str) -> list[JsonChange]:
    old_text = _run_git(repo_root, ["show", f"{baseline_ref}:{path.relative_to(repo_root).as_posix()}"])
    new_text = path.read_text(encoding="utf-8")
    return _audit_csv_values(old_text, new_text)


def _sanitize_cmd(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    target_paths = [repo_root / path for path in args.paths]
    if args.changed_only:
        target_paths = _tracked_json_paths(repo_root, args.baseline_ref)

    for path in target_paths:
        if not path.exists() and args.baseline_ref is None:
            print(f"[error] Missing JSON file: {path}", file=sys.stderr)
            return 2
        source_data = (
            _load_json_from_ref(repo_root, path.relative_to(repo_root), args.baseline_ref)
            if args.baseline_ref
            else _load_json(path)
        )
        _sanitize_file(path, repo_root, source_data)
        print(f"[ok] sanitized {path.relative_to(repo_root)}")
    return 0


def _sanitize_csv_cmd(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    target_paths = [repo_root / path for path in args.paths]
    for path in target_paths:
        if not path.exists() and args.baseline_ref is None:
            print(f"[error] Missing CSV file: {path}", file=sys.stderr)
            return 2
        source_text = (
            _run_git(repo_root, ["show", f"{args.baseline_ref}:{path.relative_to(repo_root).as_posix()}"])
            if args.baseline_ref
            else path.read_text(encoding="utf-8")
        )
        _sanitize_csv_file(path, repo_root, source_text)
        print(f"[ok] sanitized {path.relative_to(repo_root)}")
    return 0


def _audit_changes(paths: list[Path], audit_one: Any, repo_root: Path, baseline_ref: str) -> int:
    metric_changes: list[tuple[Path, JsonChange]] = []
    for path in paths:
        rel_path = path.relative_to(repo_root)
        if not _path_exists_in_ref(repo_root, rel_path, baseline_ref):
            print(f"[skip] {rel_path} (not present in baseline {baseline_ref})")
            continue
        changes = audit_one(repo_root, path, baseline_ref)
        if not changes:
            continue
        print(f"[file] {rel_path}")
        for change in changes:
            print(f"  {change.category}: {change.path}")
            if change.category == "metric-semantic":
                metric_changes.append((path, change))

    if metric_changes:
        print("[error] Metric-semantic changes detected.", file=sys.stderr)
        for path, change in metric_changes:
            print(f"  - {path.relative_to(repo_root)} :: {change.path}", file=sys.stderr)
        return 2
    return 0


def _audit_cmd(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    target_paths = [repo_root / path for path in args.paths]
    if args.changed_only:
        target_paths = _tracked_json_paths(repo_root, args.baseline_ref)
    return _audit_changes(target_paths, _audit_file, repo_root, args.baseline_ref)


def _audit_csv_cmd(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    target_paths = [repo_root / path for path in args.paths]
    if args.changed_only:
        target_paths = _tracked_csv_paths(repo_root, args.baseline_ref)
    return _audit_changes(target_paths, _audit_csv_file, repo_root, args.baseline_ref)


def _refresh_publication_hashes(publication_paths: list[PublicationPath], out_dir: Path, repo_root: Path) -> None:
    for publication in publication_paths:
        target = out_dir / publication.rel_path
        if publication.kind == "json":
            data = _load_json(target)
            _recompute_hashes(data, target, out_dir)
            _dump_json(target, data)
            continue

        source_text = target.read_text(encoding="utf-8")
        _sanitize_csv_file(target, repo_root, source_text, artifact_root=out_dir)


def _write_publication_audits(
    publication_paths: list[PublicationPath],
    repo_root: Path,
    out_dir: Path,
    audit_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    def _sanitize_audit_value(value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    return _sanitize_string(value, repo_root)
                return json.dumps(_sanitize_obj(parsed, repo_root))
            return _sanitize_string(value, repo_root)
        return _sanitize_obj(value, repo_root)

    def _audit_change_payload(change: JsonChange) -> dict[str, Any]:
        return {
            "path": change.path,
            "category": change.category,
            "old_value": _portable_json_value(_sanitize_audit_value(change.old_value)),
            "new_value": _portable_json_value(_sanitize_audit_value(change.new_value)),
        }

    json_results: list[dict[str, Any]] = []
    csv_results: list[dict[str, Any]] = []

    for publication in publication_paths:
        source_path = repo_root / publication.rel_path
        target_path = out_dir / publication.rel_path
        if publication.kind == "json":
            changes = _audit_json_values(_load_json(source_path), _load_json(target_path))
            result = {
                "path": publication.rel_path,
                "kind": "json",
                "changes": [_audit_change_payload(change) for change in changes],
            }
            json_results.append(result)
            continue

        changes = _audit_csv_values(source_path.read_text(encoding="utf-8"), target_path.read_text(encoding="utf-8"))
        result = {
            "path": publication.rel_path,
            "kind": "csv",
            "changes": [_audit_change_payload(change) for change in changes],
        }
        csv_results.append(result)

    portability_findings = _scan_portability(out_dir)
    metric_changes = [
        (item["path"], change["path"])
        for group in (*json_results, *csv_results)
        for change in group["changes"]
        if change["category"] == "metric-semantic"
        for item in [group]
    ]

    json_audit = {
        "status": "fail" if metric_changes else "ok",
        "n_files": len(json_results),
        "files": json_results,
    }
    csv_audit = {
        "status": "fail" if metric_changes else "ok",
        "n_files": len(csv_results),
        "files": csv_results,
    }
    portability_audit = {
        "status": "fail" if portability_findings else "ok",
        "root": ".",
        "findings": portability_findings,
    }

    _dump_json(audit_dir / "json_audit.json", json_audit)
    _dump_json(audit_dir / "csv_audit.json", csv_audit)
    _dump_json(audit_dir / "portability_scan.json", portability_audit)

    summary_lines = [
        "# Publication Audit Summary",
        "",
        "- publication root: `.`",
        f"- JSON files audited: `{len(json_results)}`",
        f"- CSV files audited: `{len(csv_results)}`",
        f"- metric-semantic changes: `{len(metric_changes)}`",
        f"- portability findings: `{len(portability_findings)}`",
    ]
    if metric_changes:
        summary_lines.extend(["", "## Metric-Semantic Changes", ""])
        summary_lines.extend(f"- `{path}` :: `{change_path}`" for path, change_path in metric_changes)
    if portability_findings:
        summary_lines.extend(["", "## Portability Findings", ""])
        summary_lines.extend(f"- `{finding['path']}` :: `{finding['pattern']}`" for finding in portability_findings)
    (audit_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return json_audit, csv_audit, portability_audit


def _write_publication_manifest(publication_paths: list[PublicationPath], out_dir: Path) -> None:
    manifest = {
        "publication_root": ".",
        "files": [
            {
                "path": publication.rel_path,
                "kind": publication.kind,
                "sha256": _sha256_file(out_dir / publication.rel_path),
            }
            for publication in publication_paths
        ],
        "audit": {
            "json": "audit/json_audit.json",
            "csv": "audit/csv_audit.json",
            "portability": "audit/portability_scan.json",
            "summary": "audit/summary.md",
        },
    }
    _dump_json(out_dir / "RELEASE_MANIFEST.json", manifest)


def _publish_cmd(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    audit_dir = out_dir / "audit"
    paper_path = repo_root / args.paper

    if args.paths:
        publication_paths = []
        for path in args.paths:
            if path.endswith(".json"):
                kind = "json"
            elif path.endswith(".csv"):
                kind = "csv"
            else:
                print(f"[error] Unsupported publication path (expected .json or .csv): {path}", file=sys.stderr)
                return 2
            publication_paths.append(PublicationPath(rel_path=path, kind=kind))
    else:
        publication_paths = _default_publication_paths(repo_root, paper_path)

    if not publication_paths:
        print("[error] No publication JSON/CSV paths resolved.", file=sys.stderr)
        return 2

    for publication in publication_paths:
        source = repo_root / publication.rel_path
        if not source.exists():
            print(f"[error] Missing publication source: {publication.rel_path}", file=sys.stderr)
            return 2
        target = out_dir / publication.rel_path
        if publication.kind == "json":
            _sanitize_file(target, repo_root, _load_json(source), artifact_root=out_dir)
        else:
            _sanitize_csv_file(target, repo_root, source.read_text(encoding="utf-8"), artifact_root=out_dir)

    _refresh_publication_hashes(publication_paths, out_dir, repo_root)
    _, _, portability_audit = _write_publication_audits(publication_paths, repo_root, out_dir, audit_dir)
    _write_publication_manifest(publication_paths, out_dir)

    metric_changes = []
    for audit_path in (audit_dir / "json_audit.json", audit_dir / "csv_audit.json"):
        audit = _load_json(audit_path)
        for item in audit.get("files", []):
            for change in item.get("changes", []):
                if change.get("category") == "metric-semantic":
                    metric_changes.append((item.get("path"), change.get("path")))

    if metric_changes:
        print("[error] Metric-semantic changes detected in publication copies.", file=sys.stderr)
        for rel_path, change_path in metric_changes:
            print(f"  - {rel_path} :: {change_path}", file=sys.stderr)
        return 2
    if portability_audit["status"] != "ok":
        print("[error] Portability scan found non-portable tokens in publication copies.", file=sys.stderr)
        for finding in portability_audit["findings"]:
            print(f"  - {finding['path']} :: {finding['pattern']}", file=sys.stderr)
        return 2

    print(f"[ok] wrote publication copies to {out_dir}")
    print(f"[ok] wrote audit reports to {audit_dir}")
    return 0


def _scan_portability_cmd(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    findings = _scan_portability(root)
    report = {
        "status": "fail" if findings else "ok",
        "root": ".",
        "findings": findings,
    }
    if args.out_path:
        _dump_json(Path(args.out_path).resolve(), report)
    if findings:
        print("[error] Portability findings detected.", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding['path']} :: {finding['pattern']}", file=sys.stderr)
        return 2
    print(f"[ok] No portability findings in {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sanitize, publish, and audit tracked release JSON/CSV artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sanitize = subparsers.add_parser("sanitize", help="Sanitize JSON artifacts in place.")
    sanitize.add_argument("--repo_root", default=".", help="Repository root.")
    sanitize.add_argument(
        "--baseline_ref",
        default="HEAD",
        help="Git ref used as the source payload before sanitization (default: HEAD). Use '' to sanitize working tree.",
    )
    sanitize.add_argument("--changed_only", action="store_true", help="Sanitize tracked JSON files changed vs baseline_ref.")
    sanitize.add_argument("paths", nargs="*", help="JSON paths relative to repo root.")
    sanitize.set_defaults(func=_sanitize_cmd)

    audit = subparsers.add_parser("audit", help="Classify JSON key-path changes vs a baseline git ref.")
    audit.add_argument("--repo_root", default=".", help="Repository root.")
    audit.add_argument("--baseline_ref", default="HEAD", help="Git ref used as the audit baseline.")
    audit.add_argument("--changed_only", action="store_true", help="Audit tracked JSON files changed vs baseline_ref.")
    audit.add_argument("paths", nargs="*", help="JSON paths relative to repo root.")
    audit.set_defaults(func=_audit_cmd)

    sanitize_csv = subparsers.add_parser("sanitize-csv", help="Sanitize CSV artifacts in place.")
    sanitize_csv.add_argument("--repo_root", default=".", help="Repository root.")
    sanitize_csv.add_argument(
        "--baseline_ref",
        default="HEAD",
        help="Git ref used as the source payload before sanitization (default: HEAD). Use '' to sanitize working tree.",
    )
    sanitize_csv.add_argument("paths", nargs="+", help="CSV paths relative to repo root.")
    sanitize_csv.set_defaults(func=_sanitize_csv_cmd)

    audit_csv = subparsers.add_parser("audit-csv", help="Classify CSV field changes vs a baseline git ref.")
    audit_csv.add_argument("--repo_root", default=".", help="Repository root.")
    audit_csv.add_argument("--baseline_ref", default="HEAD", help="Git ref used as the audit baseline.")
    audit_csv.add_argument("--changed_only", action="store_true", help="Audit tracked CSV files changed vs baseline_ref.")
    audit_csv.add_argument("paths", nargs="*", help="CSV paths relative to repo root.")
    audit_csv.set_defaults(func=_audit_csv_cmd)

    publish = subparsers.add_parser(
        "publish",
        help="Generate publication-safe JSON/CSV copies in a separate output tree and emit audit reports.",
    )
    publish.add_argument("--repo_root", default=".", help="Repository root.")
    publish.add_argument("--paper", default="paper/MoM_paper.md", help="Paper markdown used to resolve default public evidence paths.")
    publish.add_argument("--out_dir", default="release_public", help="Output directory for publication copies.")
    publish.add_argument("paths", nargs="*", help="Optional explicit JSON/CSV paths relative to repo root.")
    publish.set_defaults(func=_publish_cmd)

    scan = subparsers.add_parser(
        "scan-portability",
        help="Scan emitted publication artifacts for non-portable numeric tokens and machine-local paths.",
    )
    scan.add_argument("--root", default="release_public", help="Root directory to scan.")
    scan.add_argument("--out_path", default="", help="Optional JSON report path.")
    scan.set_defaults(func=_scan_portability_cmd)

    args = parser.parse_args(argv)
    if getattr(args, "baseline_ref", None) == "":
        args.baseline_ref = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
