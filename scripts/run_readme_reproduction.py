#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_readme_reproduction import CheckResult, verify_run


README_CORE_BUNDLE_REVISION = "fd571b47c1c64851e9b1989792367b9babb4af63"
README_CORE_BUNDLE_PATH = Path("clt_bundles/gemma-scope-2b-pt-res_sweep_smoke")


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class CommandRecord:
    name: str
    argv: tuple[str, ...]
    display_command: str
    started_at_utc: str
    ended_at_utc: str
    exit_code: int
    generated_files: tuple[str, ...]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _walk_files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _display_arg(arg: str, run_root: Path) -> str:
    if arg == sys.executable:
        return "python"
    path = Path(arg)
    if path.is_absolute():
        try:
            return f"$RUN_ROOT/{path.relative_to(run_root).as_posix()}"
        except ValueError:
            pass
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return arg
    return arg


def _display_command(argv: Iterable[str], run_root: Path) -> str:
    return " ".join(shlex.quote(_display_arg(arg, run_root)) for arg in argv)


def _run_one(spec: CommandSpec, *, cwd: Path, run_root: Path) -> CommandRecord:
    before = _walk_files(run_root)
    started = _utc_now_iso()
    result = subprocess.run(spec.argv, cwd=str(cwd), check=False)
    ended = _utc_now_iso()
    after = _walk_files(run_root)
    generated = tuple(sorted(after - before))
    return CommandRecord(
        name=spec.name,
        argv=tuple(_display_arg(arg, run_root) for arg in spec.argv),
        display_command=_display_command(spec.argv, run_root),
        started_at_utc=started,
        ended_at_utc=ended,
        exit_code=int(result.returncode),
        generated_files=generated,
    )


def _bundle_materialization_command(run_root: Path, *, local_files_only: bool) -> CommandSpec:
    argv = [
        sys.executable,
        "scripts/gemma_scope_to_clt.py",
        "--preset",
        "readme_core_bundle",
        "--width",
        "16k",
        "--revision",
        README_CORE_BUNDLE_REVISION,
        "--out_dir",
        str(ROOT / README_CORE_BUNDLE_PATH),
    ]
    if local_files_only:
        argv.append("--local_files_only")
    return CommandSpec(
        name="Materialize README CLT bundle",
        argv=tuple(argv),
    )


def _readme_command_specs(run_root: Path, *, local_files_only: bool) -> list[CommandSpec]:
    controls = [
        sys.executable,
        "scripts/clt_raw_comparability.py",
        "--model_name_or_path",
        "google/gemma-2-2b",
        "--disamb_path",
        "data/disamb_pairs.jsonl",
        "--clt_repo",
        str(ROOT / README_CORE_BUNDLE_PATH),
        "--layers",
        "4,8,12",
        "--seed",
        "42",
        "--bootstrap_n",
        "1000",
        "--bootstrap_seed",
        "42",
        "--no-hard_fail_primary_logodds",
        "--primary_logodds_residual_tol",
        "5e-06",
        "--run_pca_baseline",
        "--run_random_orth_baseline",
        "--run_faithfulness_decomposition_arms",
        "--out_csv",
        str(run_root / "clt_raw_comparability_l4_l8_l12_controls_full.csv"),
        "--out_json",
        str(run_root / "clt_raw_comparability_l4_l8_l12_controls_full.summary.json"),
    ]
    if local_files_only:
        controls.append("--local_files_only")

    f32 = [
        sys.executable,
        "scripts/clt_raw_comparability.py",
        "--model_name_or_path",
        "google/gemma-2-2b",
        "--disamb_path",
        "data/disamb_pairs.jsonl",
        "--clt_repo",
        str(ROOT / README_CORE_BUNDLE_PATH),
        "--layers",
        "4,8,12",
        "--device",
        "cpu",
        "--torch_dtype",
        "float32",
        "--seed",
        "42",
        "--bootstrap_n",
        "5000",
        "--bootstrap_seed",
        "42",
        "--no-hard_fail_primary_logodds",
        "--primary_logodds_residual_tol",
        "5e-06",
        "--out_csv",
        str(run_root / "r1_full_cpu_f32.csv"),
        "--out_json",
        str(run_root / "r1_full_cpu_f32.summary.json"),
    ]
    if local_files_only:
        f32.append("--local_files_only")

    f64 = [
        sys.executable,
        "scripts/clt_raw_comparability.py",
        "--model_name_or_path",
        "google/gemma-2-2b",
        "--disamb_path",
        "data/disamb_pairs.jsonl",
        "--clt_repo",
        str(ROOT / README_CORE_BUNDLE_PATH),
        "--layers",
        "4,8,12",
        "--device",
        "cpu",
        "--torch_dtype",
        "float64",
        "--seed",
        "42",
        "--bootstrap_n",
        "5000",
        "--bootstrap_seed",
        "42",
        "--no-hard_fail_primary_logodds",
        "--primary_logodds_residual_tol",
        "5e-06",
        "--out_csv",
        str(run_root / "r1_full_cpu_f64.csv"),
        "--out_json",
        str(run_root / "r1_full_cpu_f64.summary.json"),
    ]
    if local_files_only:
        f64.append("--local_files_only")

    return [
        CommandSpec(
            name="DISAMB matched controls",
            argv=tuple(controls),
        ),
        CommandSpec(
            name="DISAMB strict CPU rerun (float32)",
            argv=tuple(f32),
        ),
        CommandSpec(
            name="DISAMB strict CPU rerun (float64)",
            argv=tuple(f64),
        ),
        CommandSpec(
            name="Endpoint decomposition (float64)",
            argv=(
                sys.executable,
                "scripts/mom_endpoint_decomp_analyze.py",
                "--comparability_csv",
                str(run_root / "r1_full_cpu_f64.csv"),
                "--comparability_summary",
                str(run_root / "r1_full_cpu_f64.summary.json"),
                "--disamb_path",
                "data/disamb_pairs.jsonl",
                "--tokenizer_name_or_path",
                "google/gemma-2-2b",
                "--bootstrap_n",
                "5000",
                "--ci",
                "0.95",
                "--seed",
                "42",
                "--primary_residual_tol",
                "5e-06",
                "--out_pair_csv",
                str(run_root / "r1_full_cpu_f64.endpoint_pair_aggregates_v2.csv"),
                "--out_json",
                str(run_root / "r1_full_cpu_f64.endpoint_decomp_summary_v2.json"),
                "--out_md",
                str(run_root / "r1_full_cpu_f64.endpoint_decomp_summary_v2.md"),
            ),
        ),
        CommandSpec(
            name="CF task-axis comparability",
            argv=(
                sys.executable,
                "scripts/clt_raw_comparability_cf.py",
                "--model_name_or_path",
                "google/gemma-2-2b",
                "--cf_path",
                "data/counterfactual.jsonl",
                "--coh_path",
                "data/coherence.jsonl",
                "--clt_repo",
                str(ROOT / README_CORE_BUNDLE_PATH),
                "--layers",
                "4,8,12",
                "--device",
                "cpu",
                "--torch_dtype",
                "float32",
                "--seed",
                "42",
                "--bootstrap_n",
                "1000",
                "--bootstrap_seed",
                "42",
                "--out_csv",
                str(run_root / "clt_raw_comparability_cf_l4_l8_l12_final_f32.csv"),
                "--out_json",
                str(run_root / "clt_raw_comparability_cf_l4_l8_l12_final_f32.summary.json"),
            ),
        ),
        CommandSpec(
            name="COH task-axis comparability",
            argv=(
                sys.executable,
                "scripts/clt_raw_comparability_coh.py",
                "--model_name_or_path",
                "google/gemma-2-2b",
                "--cf_path",
                "data/counterfactual.jsonl",
                "--coh_path",
                "data/coherence.jsonl",
                "--clt_repo",
                str(ROOT / README_CORE_BUNDLE_PATH),
                "--layers",
                "4,8,12",
                "--device",
                "cpu",
                "--torch_dtype",
                "float32",
                "--seed",
                "42",
                "--bootstrap_n",
                "1000",
                "--bootstrap_seed",
                "42",
                "--out_csv",
                str(run_root / "clt_raw_comparability_coh_l4_l8_l12_final_f32.csv"),
                "--out_json",
                str(run_root / "clt_raw_comparability_coh_l4_l8_l12_final_f32.summary.json"),
            ),
        ),
    ]


def _render_check(result: CheckResult) -> str:
    detail = f"observed={result.observed!r}; expected={result.expected!r}"
    if result.note:
        detail += f"; {result.note}"
    status = "PASS" if result.status == "pass" else "FAIL"
    return f"- {status}: `{result.name}` ({detail}; reference=`{result.reference_path}`)"


def _render_report(
    *,
    run_root: Path,
    created_run_root: bool,
    records: list[CommandRecord],
    missing: list[Path],
    checks: list[CheckResult],
) -> str:
    lines = [
        "# README Reproduction Report",
        "",
        f"- run root: `$RUN_ROOT` (`{run_root.name}`)",
        f"- run root created by script: `{'yes' if created_run_root else 'no'}`",
        f"- repository commit: `{subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()}`",
    ]

    lines.extend(["", "## Command Log", ""])
    for index, record in enumerate(records, start=1):
        status = "PASS" if record.exit_code == 0 else "FAIL"
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

    if missing:
        lines.extend(["## Missing Outputs", ""])
        for path in missing:
            lines.append(f"- `$RUN_ROOT/{path.name}`")
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.extend(["## Output Files", ""])
    for path in sorted(path.relative_to(run_root).as_posix() for path in run_root.rglob("*") if path.is_file()):
        lines.append(f"- `$RUN_ROOT/{path}`")

    lines.extend(["", "## Numeric Checks", ""])
    for check in checks:
        lines.append(_render_check(check))

    overall = "PASS"
    if any(record.exit_code != 0 for record in records) or any(check.status != "pass" for check in checks):
        overall = "FAIL"
    lines.extend(["", "## Overall", "", f"- status: `{overall}`"])
    return "\n".join(lines) + "\n"


def _write_json_log(path: Path, records: list[CommandRecord], checks: list[CheckResult]) -> None:
    payload = {
        "records": [asdict(record) for record in records],
        "checks": [asdict(check) for check in checks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute the README core-claims reproduction path, verify the outputs, and write a report."
    )
    parser.add_argument(
        "--run_root",
        default="",
        help="Optional run directory. Defaults to a fresh temp directory outside the repo.",
    )
    parser.add_argument(
        "--report_path",
        default="reports/readme_reproduction_report.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--json_log_path",
        default="reports/readme_reproduction_log.json",
        help="Machine-readable command/check log path.",
    )
    parser.add_argument(
        "--skip_bundle_materialization",
        action="store_true",
        help="Do not materialize the README CLT bundle even if it is missing.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Use only local HF cache for bundle materialization and DISAMB comparability runs.",
    )
    args = parser.parse_args(argv)

    created_run_root = False
    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
    else:
        run_root = Path(tempfile.mkdtemp(prefix="mom_core_"))
        created_run_root = True

    if any(run_root.iterdir()):
        print(f"[error] Run root must start empty: {run_root}", file=sys.stderr)
        return 2

    records: list[CommandRecord] = []
    bundle_path = ROOT / README_CORE_BUNDLE_PATH
    if not bundle_path.exists():
        if args.skip_bundle_materialization:
            print(f"[error] Missing README CLT bundle: {bundle_path}", file=sys.stderr)
            return 2
        record = _run_one(
            _bundle_materialization_command(run_root, local_files_only=bool(args.local_files_only)),
            cwd=ROOT,
            run_root=run_root,
        )
        records.append(record)
        if record.exit_code != 0:
            report = _render_report(
                run_root=run_root,
                created_run_root=created_run_root,
                records=records,
                missing=[],
                checks=[],
            )
            report_path = Path(args.report_path).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report, encoding="utf-8")
            _write_json_log(Path(args.json_log_path).expanduser().resolve(), records, [])
            return 2

    for spec in _readme_command_specs(run_root, local_files_only=bool(args.local_files_only)):
        record = _run_one(spec, cwd=ROOT, run_root=run_root)
        records.append(record)
        if record.exit_code != 0:
            break

    missing, checks = verify_run(run_root)
    report = _render_report(
        run_root=run_root,
        created_run_root=created_run_root,
        records=records,
        missing=missing,
        checks=checks,
    )

    report_path = Path(args.report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    _write_json_log(Path(args.json_log_path).expanduser().resolve(), records, checks)

    if any(record.exit_code != 0 for record in records):
        print(report, file=sys.stderr)
        return 2
    if missing or any(check.status != "pass" for check in checks):
        print(report, file=sys.stderr)
        return 2

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
