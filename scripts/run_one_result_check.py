#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.utils import get_best_device
from scripts.paper_requirements import PAPER_CLT_BUNDLE_PATH, PAPER_MODEL_REPO_ID
from scripts.run_readme_reproduction import (
    CommandRecord,
    CommandSpec,
    _bundle_materialization_command,
    _display_arg,
    _display_command,
    _render_check,
    _utc_now_iso,
    _walk_files,
)
from scripts.verify_one_result_check import verify_run
from scripts.verify_readme_reproduction import _is_failure_status


def _run_one(spec: CommandSpec, *, cwd: Path, run_root: Path, dry_run: bool) -> CommandRecord:
    before = _walk_files(run_root)
    started = _utc_now_iso()
    exit_code = 0
    if dry_run:
        print(_display_command(spec.argv, run_root))
    else:
        result = subprocess.run(spec.argv, cwd=str(cwd), check=False)
        exit_code = int(result.returncode)
    ended = _utc_now_iso()
    after = _walk_files(run_root)
    generated = tuple(sorted(after - before))
    return CommandRecord(
        name=spec.name,
        argv=tuple(_display_arg(arg, run_root) for arg in spec.argv),
        display_command=_display_command(spec.argv, run_root),
        started_at_utc=started,
        ended_at_utc=ended,
        exit_code=exit_code,
        generated_files=generated,
        artifact_kind=spec.artifact_kind,
    )


def _device_available(device: str) -> bool:
    import torch

    kind = str(device).lower()
    if kind == "cpu":
        return True
    if kind == "cuda":
        return bool(torch.cuda.is_available())
    if kind == "mps":
        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return False


def _resolve_requested_device(device: str) -> str:
    requested = str(device).strip().lower()
    if requested == "auto":
        return str(get_best_device().type)
    return requested


def _one_result_command_spec(run_root: Path, *, local_files_only: bool, device: str) -> CommandSpec:
    argv = [
        sys.executable,
        "scripts/clt_raw_comparability.py",
        "--model_name_or_path",
        PAPER_MODEL_REPO_ID,
        "--disamb_path",
        "data/disamb_pairs.jsonl",
        "--clt_repo",
        str(ROOT / PAPER_CLT_BUNDLE_PATH),
        "--layers",
        "4",
        "--device",
        str(device),
        "--torch_dtype",
        "float32",
        "--seed",
        "42",
        "--bootstrap_n",
        "200",
        "--bootstrap_seed",
        "42",
        "--no-hard_fail_primary_logodds",
        "--primary_logodds_residual_tol",
        "5e-06",
        "--run_pca_baseline",
        "--run_random_orth_baseline",
        "--run_faithfulness_decomposition_arms",
        "--out_csv",
        str(run_root / "one_result_controls_l4.csv"),
        "--out_json",
        str(run_root / "one_result_controls_l4.summary.json"),
    ]
    if local_files_only:
        argv.append("--local_files_only")
    return CommandSpec(name="Layer-4 controls quick result check", argv=tuple(argv))


def _render_report(
    *,
    run_root: Path,
    created_run_root: bool,
    records: list[CommandRecord],
    missing: list[Path],
    checks: list,
    dry_run: bool,
) -> str:
    lines = [
        "# One Result Check Report",
        "",
        f"- run root: `$RUN_ROOT` (`{run_root.name}`)",
        f"- run root created by script: `{'yes' if created_run_root else 'no'}`",
        f"- repository commit: `{subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()}`",
        f"- mode: `{'dry_run' if dry_run else 'execute'}`",
        "",
        "## Command Log",
        "",
    ]

    for index, record in enumerate(records, start=1):
        status = "PLAN_ONLY" if dry_run else ("PASS" if record.exit_code == 0 else "FAIL")
        lines.append(f"### {index}. {record.name}")
        lines.append("")
        lines.append(f"- status: `{status}`")
        lines.append(f"- started_at_utc: `{record.started_at_utc}`")
        lines.append(f"- ended_at_utc: `{record.ended_at_utc}`")
        lines.append(f"- exit_code: `{record.exit_code}`")
        lines.append("- command:")
        lines.append("")
        lines.append("```bash")
        lines.append(record.display_command)
        lines.append("```")
        lines.append("")
        if record.generated_files:
            lines.append("- generated files:")
            for rel_path in record.generated_files:
                lines.append(f"  - `$RUN_ROOT/{rel_path}`")
        else:
            lines.append("- generated files: `none`")
        lines.append("")

    if dry_run:
        lines.extend(["## Overall", "", "- overall_status: `PLAN_ONLY`"])
        return "\n".join(lines) + "\n"

    if missing:
        lines.extend(["## Missing Outputs", ""])
        for path in missing:
            lines.append(f"- `$RUN_ROOT/{path.relative_to(run_root).as_posix()}`")
        lines.extend(["", "## Overall", "", "- overall_status: `FAIL`"])
        return "\n".join(lines) + "\n"

    lines.extend(["## Output Files", ""])
    for path in sorted(path.relative_to(run_root).as_posix() for path in run_root.rglob("*") if path.is_file()):
        lines.append(f"- `$RUN_ROOT/{path}`")

    lines.extend(["", "## Numeric Checks", ""])
    for check in checks:
        lines.append(_render_check(check))

    overall = "PASS"
    if any(record.exit_code != 0 for record in records) or any(_is_failure_status(check.status) for check in checks):
        overall = "FAIL"
    lines.extend(["", "## Overall", "", f"- overall_status: `{overall}`"])
    return "\n".join(lines) + "\n"


def _write_json_log(path: Path, *, records: list[CommandRecord], missing: list[Path], checks: list) -> None:
    payload = {
        "records": [asdict(record) for record in records],
        "missing": [str(path) for path in missing],
        "checks": [asdict(check) for check in checks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute a fast single-result MoM verification and write a report."
    )
    parser.add_argument(
        "--run_root",
        default="",
        help="Optional run directory. Defaults to a fresh temp directory outside the repo.",
    )
    parser.add_argument(
        "--report_path",
        default="",
        help="Markdown report path. Defaults to `$RUN_ROOT/one_result_check_report.md`.",
    )
    parser.add_argument(
        "--json_log_path",
        default="",
        help="Machine-readable command/check log path. Defaults to `$RUN_ROOT/one_result_check_log.json`.",
    )
    parser.add_argument(
        "--skip_bundle_materialization",
        action="store_true",
        help="Do not materialize the CLT bundle even if it is missing.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Use only local HF cache for bundle materialization and the quick result check.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device for the quick result check. Default is cpu.",
    )
    parser.add_argument(
        "--require_accelerator",
        action="store_true",
        help="Fail unless the resolved device is a GPU accelerator (CUDA or MPS).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print and log the exact commands without executing them.",
    )
    args = parser.parse_args(argv)

    resolved_device = _resolve_requested_device(str(args.device))
    if not _device_available(resolved_device):
        print(f"[error] Requested device is unavailable: {resolved_device}", file=sys.stderr)
        return 2
    if bool(args.require_accelerator) and resolved_device == "cpu":
        print("[error] No accelerator available; resolved device is cpu.", file=sys.stderr)
        return 2

    created_run_root = False
    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
    else:
        run_root = Path(tempfile.mkdtemp(prefix="mom_one_result_"))
        created_run_root = True

    if any(run_root.iterdir()):
        print(f"[error] Run root must start empty: {run_root}", file=sys.stderr)
        return 2

    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else run_root / "one_result_check_report.md"
    json_log_path = Path(args.json_log_path).expanduser().resolve() if args.json_log_path else run_root / "one_result_check_log.json"

    records: list[CommandRecord] = []
    bundle_path = ROOT / PAPER_CLT_BUNDLE_PATH
    if not bundle_path.exists():
        if args.skip_bundle_materialization:
            print(f"[error] Missing CLT bundle: {bundle_path}", file=sys.stderr)
            return 2
        record = _run_one(
            _bundle_materialization_command(run_root, local_files_only=bool(args.local_files_only)),
            cwd=ROOT,
            run_root=run_root,
            dry_run=bool(args.dry_run),
        )
        records.append(record)
        if record.exit_code != 0:
            report = _render_report(
                run_root=run_root,
                created_run_root=created_run_root,
                records=records,
                missing=[],
                checks=[],
                dry_run=bool(args.dry_run),
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report, encoding="utf-8")
            _write_json_log(json_log_path, records=records, missing=[], checks=[])
            print(report, file=sys.stderr)
            return 2

    record = _run_one(
        _one_result_command_spec(
            run_root,
            local_files_only=bool(args.local_files_only),
            device=resolved_device,
        ),
        cwd=ROOT,
        run_root=run_root,
        dry_run=bool(args.dry_run),
    )
    records.append(record)

    missing: list[Path] = []
    checks = []
    if not args.dry_run and record.exit_code == 0:
        _write_json_log(json_log_path, records=records, missing=[], checks=[])
        missing_artifacts, checks = verify_run(run_root)
        missing = [artifact.path for artifact in missing_artifacts]

    report = _render_report(
        run_root=run_root,
        created_run_root=created_run_root,
        records=records,
        missing=missing,
        checks=checks,
        dry_run=bool(args.dry_run),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    _write_json_log(json_log_path, records=records, missing=missing, checks=checks)

    stream = sys.stdout if (args.dry_run or (record.exit_code == 0 and not missing and not any(_is_failure_status(check.status) for check in checks))) else sys.stderr
    print(report, file=stream)

    if args.dry_run:
        return 0
    if record.exit_code != 0 or missing or any(_is_failure_status(check.status) for check in checks):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
