#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StepResult:
    name: str
    status: str
    detail: str


def _run(argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True, check=False)


def _ok(detail: str) -> StepResult:
    return StepResult(name="", status="ok", detail=detail)


def _check_citation(citation_path: Path) -> StepResult:
    if not citation_path.is_file():
        return StepResult("citation", "fail", f"Missing {citation_path.relative_to(ROOT)}")
    payload = yaml.safe_load(citation_path.read_text(encoding="utf-8"))
    required = ("cff-version", "message", "title", "authors", "license", "preferred-citation")
    missing = [field for field in required if not payload.get(field)]
    if missing:
        return StepResult("citation", "fail", f"CITATION.cff missing required field(s): {', '.join(missing)}")
    preferred = payload["preferred-citation"]
    preferred_missing = [
        field
        for field in ("type", "title", "year", "authors")
        if not preferred.get(field)
    ]
    if preferred_missing:
        return StepResult(
            "citation",
            "fail",
            f"CITATION.cff preferred-citation missing field(s): {', '.join(preferred_missing)}",
        )
    return StepResult("citation", "ok", "CITATION.cff parsed and required fields are present")


def _check_archive_metadata(zenodo_path: Path, citation_path: Path) -> StepResult:
    if not zenodo_path.is_file():
        return StepResult("archive-metadata", "fail", f"Missing {zenodo_path.relative_to(ROOT)}")
    payload = json.loads(zenodo_path.read_text(encoding="utf-8"))
    required = ("title", "upload_type", "description", "creators", "license")
    missing = [field for field in required if not payload.get(field)]
    if missing:
        return StepResult("archive-metadata", "fail", f".zenodo.json missing field(s): {', '.join(missing)}")
    citation = yaml.safe_load(citation_path.read_text(encoding="utf-8"))
    citation_license = str(citation.get("license", "")).strip()
    zenodo_license = str(payload.get("license", "")).strip()
    if citation_license and zenodo_license and citation_license != zenodo_license:
        return StepResult(
            "archive-metadata",
            "fail",
            f"License mismatch between CITATION.cff ({citation_license}) and .zenodo.json ({zenodo_license})",
        )
    return StepResult("archive-metadata", "ok", ".zenodo.json parsed and matches citation license")


def _check_subprocess(name: str, argv: list[str], *, cwd: Path) -> StepResult:
    result = _run(argv, cwd=cwd)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit_code={result.returncode}"
        return StepResult(name, "fail", detail)
    summary = result.stdout.strip().splitlines()
    detail = summary[-1] if summary else "ok"
    return StepResult(name, "ok", detail)


def _check_bundle_includes_citation(
    publication_root: Path,
    archive_path: Path,
) -> StepResult:
    result = _run(
        [
            sys.executable,
            "scripts/build_minimal_publish_bundle.py",
            "--dry_run",
            "--archive",
            str(archive_path),
            "--publication_root",
            str(publication_root),
        ],
        cwd=ROOT,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit_code={result.returncode}"
        return StepResult("bundle-dry-run", "fail", detail)
    lines = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    required = {"CITATION.cff", "LICENSE", "README.md"}
    missing = sorted(required - lines)
    if missing:
        return StepResult("bundle-dry-run", "fail", f"Dry-run bundle omitted required file(s): {', '.join(missing)}")
    if ".zenodo.json" not in lines:
        return StepResult("bundle-dry-run", "fail", "Dry-run bundle omitted .zenodo.json")
    return StepResult("bundle-dry-run", "ok", "Minimal publish bundle dry-run includes citation and archive metadata")


def _render_summary(results: list[StepResult]) -> str:
    lines = ["# Archival Readiness Check", ""]
    for result in results:
        lines.append(f"- {result.status.upper()}: `{result.name}` - {result.detail}")
    return "\n".join(lines) + "\n"


def _sanitize_detail(detail: str, publication_root: Path) -> str:
    sanitized = str(detail)
    replacements = {
        str(publication_root / "audit"): "$PUBLICATION_ROOT/audit",
        f"/private{publication_root / 'audit'}": "$PUBLICATION_ROOT/audit",
        str(publication_root): "$PUBLICATION_ROOT",
        f"/private{publication_root}": "$PUBLICATION_ROOT",
        str(ROOT): ".",
    }
    for source, target in replacements.items():
        sanitized = sanitized.replace(source, target)
    sanitized = sanitized.replace("/private$PUBLICATION_ROOT/audit", "$PUBLICATION_ROOT/audit")
    sanitized = sanitized.replace("/private$PUBLICATION_ROOT", "$PUBLICATION_ROOT")
    return sanitized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the structural archival-readiness checks.")
    parser.add_argument(
        "--publication_root",
        default="",
        help="Optional output directory for publication-safe copies. Defaults to a temp directory.",
    )
    parser.add_argument(
        "--skip_pytest",
        action="store_true",
        help="Skip `python -m pytest -q`.",
    )
    parser.add_argument(
        "--skip_readme_verifier",
        action="store_true",
        help="Skip verifying an existing README reproduction run.",
    )
    parser.add_argument(
        "--readme_run_root",
        default="",
        help="Existing README reproduction output directory to verify.",
    )
    parser.add_argument(
        "--readme_report_path",
        default="reports/readme_reproduction_report.md",
        help="Report path passed to the README verifier when --readme_run_root is set.",
    )
    parser.add_argument(
        "--report_path",
        default="reports/archive_readiness_check.md",
        help="Markdown summary report path.",
    )
    args = parser.parse_args(argv)

    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.publication_root:
        publication_root = Path(args.publication_root).expanduser().resolve()
        publication_root.mkdir(parents=True, exist_ok=True)
        if any(publication_root.iterdir()):
            print(f"[error] --publication_root must start empty: {publication_root}", file=sys.stderr)
            return 2
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="archival_readiness_")
        publication_root = Path(temp_dir_obj.name) / "release_public"

    archive_path = publication_root.parent / "mom_minimal_publish_bundle.tar.gz"
    report_results: list[StepResult] = []

    citation_path = ROOT / "CITATION.cff"
    zenodo_path = ROOT / ".zenodo.json"
    report_results.append(_check_citation(citation_path))
    report_results.append(_check_archive_metadata(zenodo_path, citation_path))
    report_results.append(
        _check_subprocess("evidence-contract", [sys.executable, "scripts/check_evidence_contract.py"], cwd=ROOT)
    )
    report_results.append(
        _check_subprocess(
            "evidence-contract-fields",
            [sys.executable, "scripts/check_evidence_contract_fields.py"],
            cwd=ROOT,
        )
    )
    if not args.skip_pytest:
        report_results.append(_check_subprocess("pytest", [sys.executable, "-m", "pytest", "-q"], cwd=ROOT))

    publish_result = _check_subprocess(
        "publish-artifacts",
        [
            sys.executable,
            "scripts/release_json_artifacts.py",
            "publish",
            "--out_dir",
            str(publication_root),
        ],
        cwd=ROOT,
    )
    report_results.append(publish_result)
    if publish_result.status == "ok":
        report_results.append(
            _check_subprocess(
                "portability-scan",
                [
                    sys.executable,
                    "scripts/release_json_artifacts.py",
                    "scan-portability",
                    "--root",
                    str(publication_root),
                    "--out_path",
                    str(publication_root / "audit" / "portability_scan.json"),
                ],
                cwd=ROOT,
            )
        )
        report_results.append(_check_bundle_includes_citation(publication_root, archive_path))

    if args.skip_readme_verifier:
        report_results.append(
            StepResult("readme-verifier", "skip", "Skipped by --skip_readme_verifier")
        )
    elif args.readme_run_root:
        report_results.append(
            _check_subprocess(
                "readme-verifier",
                [
                    sys.executable,
                    "scripts/verify_readme_reproduction.py",
                    "--run_root",
                    str(Path(args.readme_run_root).expanduser().resolve()),
                    "--report_path",
                    str(Path(args.readme_report_path).expanduser().resolve()),
                ],
                cwd=ROOT,
            )
        )
    else:
        report_results.append(
            StepResult(
                "readme-verifier",
                "skip",
                "No --readme_run_root provided; skipping heavy README verification",
            )
        )

    report_results = [
        StepResult(result.name, result.status, _sanitize_detail(result.detail, publication_root))
        for result in report_results
    ]
    summary = _render_summary(report_results)
    report_path = Path(args.report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(summary, encoding="utf-8")

    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()

    failing = [result for result in report_results if result.status == "fail"]
    if failing:
        print(summary, file=sys.stderr)
        return 2

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
