#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_requirements import (
    PAPER_CLT_BUNDLE_PATH as README_CORE_BUNDLE_PATH,
    PAPER_MODEL_REPO_ID as MODEL_REPO_ID,
    PAPER_MODEL_REVISION as MODEL_REVISION,
    PAPER_SCOPE_REPO_ID as SCOPE_REPO_ID,
    PAPER_SCOPE_REVISION as README_CORE_BUNDLE_REVISION,
)
from scripts.reviewer_assets import local_asset_results as _local_asset_probe_results

SUCCESS = "PASS"
FAILURE = "FAIL"
TIMEOUT = "TIMEOUT"
SMOKE_TIMEOUT_SECONDS = 180
PAPER_DRY_RUN_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class StepResult:
    name: str
    status: str
    detail: str


def _display_command(argv: Sequence[str]) -> str:
    return shlex.join(str(x) for x in argv)


def _write_log(log_dir: Path, stem: str, payload: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / stem
    path.write_text(payload, encoding="utf-8")
    return path


def _coerce_output(payload: str | bytes | None) -> str:
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return str(payload)


def _tail_snippet(*, stdout: str, stderr: str) -> str:
    tail = (stderr or stdout or "").strip().splitlines()[-8:]
    snippet = " | ".join(line.strip() for line in tail if line.strip())
    if not snippet:
        snippet = "no stderr/stdout captured"
    return snippet


def _timeout_default(env_name: str, fallback: int) -> int:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return int(fallback)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be an integer if set; got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{env_name} must be positive if set; got {value}")
    return value


def _run_step(
    *,
    name: str,
    argv: Sequence[str],
    cwd: Path,
    log_dir: Path,
    timeout_seconds: int | None = None,
) -> StepResult:
    try:
        proc = subprocess.run(
            list(argv),
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_output(exc.stdout)
        stderr = _coerce_output(exc.stderr)
        stdout_path = _write_log(log_dir, f"{name}.stdout.log", stdout)
        stderr_path = _write_log(log_dir, f"{name}.stderr.log", stderr)
        return StepResult(
            name=name,
            status=TIMEOUT,
            detail=(
                f"command timed out after {int(timeout_seconds or 0)}s; "
                f"cmd={_display_command(argv)}; tail={_tail_snippet(stdout=stdout, stderr=stderr)}; "
                f"logs: {stdout_path.name}, {stderr_path.name}"
            ),
        )

    stdout_path = _write_log(log_dir, f"{name}.stdout.log", proc.stdout)
    stderr_path = _write_log(log_dir, f"{name}.stderr.log", proc.stderr)
    if int(proc.returncode) == 0:
        return StepResult(name=name, status=SUCCESS, detail=f"command succeeded; logs: {stdout_path.name}, {stderr_path.name}")
    return StepResult(
        name=name,
        status=FAILURE,
        detail=(
            f"command failed with exit_code={int(proc.returncode)}; "
            f"cmd={_display_command(argv)}; tail={_tail_snippet(stdout=proc.stdout, stderr=proc.stderr)}; "
            f"logs: {stdout_path.name}, {stderr_path.name}"
        ),
    )


def _smoke_command(results_dir: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        str(ROOT / "scripts" / "run_paper.py"),
        "smoke",
        "--results_dir",
        str(results_dir),
        "--skip_dataset",
    )


def _paper_dry_run_command(run_root: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        str(ROOT / "scripts" / "run_mom_paper.py"),
        "--dry_run",
        "--run_root",
        str(run_root),
        "--local_files_only",
    )


def _local_asset_results() -> list[StepResult]:
    return [
        StepResult(name=result.name, status=SUCCESS if result.ok else FAILURE, detail=result.detail)
        for result in _local_asset_probe_results()
    ]


def _best_accelerator() -> str:
    try:
        import torch
    except Exception:
        return ""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return ""


def _next_commands_snippet(results: Sequence[StepResult] | None = None) -> str:
    if results is not None and any(result.name.startswith("local_") and result.status != SUCCESS for result in results):
        return "\n".join(
            [
                "make reviewer-assets",
                "make reviewer-check",
            ]
        )
    commands = ["make one-result-check"]
    if _best_accelerator():
        commands.append("make one-result-check-gpu")
    commands.append('make reproduction MOM_PAPER_ARGS="--run_root /tmp/mom_paper_review_run"')
    return "\n".join(commands)


def _render_summary(*, quickcheck_root: Path, results: Sequence[StepResult]) -> str:
    ready = all(result.status == SUCCESS for result in results)
    lines = [
        "# MoM Reviewer Quick Check",
        "",
        f"- quickcheck_root: `{quickcheck_root}`",
    ]
    for result in results:
        lines.append(f"- {result.name}: `{result.status}`; {result.detail}")
    lines.extend(
        [
            f"- ready_for_offline_paper_run: `{'PASS' if ready else 'FAIL'}`",
            "",
            "## Next Commands",
            "",
            "```bash",
            _next_commands_snippet(results),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a fast reviewer readiness check: offline smoke, paper-runner dry-run, and local asset validation."
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the quickcheck temp directory after a successful run.",
    )
    parser.add_argument(
        "--smoke-timeout-seconds",
        type=int,
        default=_timeout_default("MOM_REVIEWER_SMOKE_TIMEOUT_SECONDS", SMOKE_TIMEOUT_SECONDS),
        help="Timeout for the offline smoke step in seconds.",
    )
    parser.add_argument(
        "--paper-dry-run-timeout-seconds",
        type=int,
        default=_timeout_default("MOM_REVIEWER_DRY_RUN_TIMEOUT_SECONDS", PAPER_DRY_RUN_TIMEOUT_SECONDS),
        help="Timeout for the paper dry-run step in seconds.",
    )
    args = parser.parse_args(argv)
    if int(args.smoke_timeout_seconds) <= 0:
        parser.error("--smoke-timeout-seconds must be positive")
    if int(args.paper_dry_run_timeout_seconds) <= 0:
        parser.error("--paper-dry-run-timeout-seconds must be positive")

    quickcheck_root = Path(tempfile.mkdtemp(prefix="mom_reviewer_check_"))
    smoke_results_dir = quickcheck_root / "smoke_results"
    paper_dry_run_root = quickcheck_root / "paper_dry_run"
    log_dir = quickcheck_root / "logs"

    print(f"[1/3] Offline smoke: {_display_command(_smoke_command(smoke_results_dir))}")
    results = [
        _run_step(
            name="offline_smoke",
            argv=_smoke_command(smoke_results_dir),
            cwd=ROOT,
            log_dir=log_dir,
            timeout_seconds=int(args.smoke_timeout_seconds),
        )
    ]

    print(f"[2/3] Paper dry-run: {_display_command(_paper_dry_run_command(paper_dry_run_root))}")
    results.append(
        _run_step(
            name="paper_dry_run",
            argv=_paper_dry_run_command(paper_dry_run_root),
            cwd=ROOT,
            log_dir=log_dir,
            timeout_seconds=int(args.paper_dry_run_timeout_seconds),
        )
    )

    print("[3/3] Local paper assets: checking cached model, CLT bundle/source, and fixed-layer SAE support")
    results.extend(_local_asset_results())

    summary = _render_summary(quickcheck_root=quickcheck_root, results=results)
    print(summary)

    ready = all(result.status == SUCCESS for result in results)
    if ready and bool(args.cleanup):
        shutil.rmtree(quickcheck_root, ignore_errors=True)
    elif not ready:
        print(f"[info] quickcheck artifacts kept at {quickcheck_root}", file=sys.stderr)
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
