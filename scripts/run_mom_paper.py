#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_requirements import (
    PAPER_CLT_BUNDLE_PATH,
    PAPER_FIXED_LAYER_SAE_LAYER,
    PAPER_FIXED_LAYER_SAE_RUN_NAME,
    PAPER_MODEL_REPO_ID,
    PAPER_MODEL_REVISION,
    PAPER_SCOPE_REPO_ID,
    PAPER_SCOPE_REVISION,
    PAPER_SIX_LAYER_PROFILE_LAYERS,
)
from scripts.run_readme_reproduction import (
    CommandRecord,
    CommandSpec,
    _bundle_materialization_command,
    _display_arg,
    _display_command,
    _readme_command_specs,
    _render_check,
    _utc_now_iso,
    _walk_files,
    _write_json_log,
)
from scripts.verify_mom_paper import _display_reference_path, _load_result_row, verify_run
from scripts.verify_readme_reproduction import _is_failure_status


MODEL_REVISION = PAPER_MODEL_REVISION
SAE_REVISION = PAPER_SCOPE_REVISION
README_CORE_BUNDLE_PATH = PAPER_CLT_BUNDLE_PATH
SIX_LAYER_PROFILE_LAYERS = ",".join(str(layer) for layer in PAPER_SIX_LAYER_PROFILE_LAYERS)
FIXED_LAYER_SAE_RUN_NAME = PAPER_FIXED_LAYER_SAE_RUN_NAME
FIXED_LAYER_SAE_LAYER = str(PAPER_FIXED_LAYER_SAE_LAYER)
PAPER_SUPPORT_DEVICE = "cpu"
SUPPORT_MANIFEST_SPECS = (
    ("six-layer raw", "paper_support/gemma2b_raw_6layer_full_seed42.manifest.json"),
    ("six-layer clt", "paper_support/gemma2b_clt_6layer_full_seed42.manifest.json"),
    ("fixed-layer sae", "paper_support/gemma2b_sae.manifest.json"),
)


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


def _paper_support_command_specs(run_root: Path, *, local_files_only: bool) -> list[CommandSpec]:
    support_root = run_root / "paper_support"

    raw = [
        sys.executable,
        "aom_eval.py",
        "--model_name_or_path",
        PAPER_MODEL_REPO_ID,
        "--revision",
        MODEL_REVISION,
        "--device",
        PAPER_SUPPORT_DEVICE,
        "--attn_implementation",
        "eager",
        "--disamb_path",
        "data/disamb_pairs.jsonl",
        "--cf_path",
        "data/counterfactual.jsonl",
        "--coh_path",
        "data/coherence.jsonl",
        "--bootstrap_n",
        "1000",
        "--bootstrap_seed",
        "42",
        "--ci",
        "0.95",
        "--seed",
        "42",
        "--run_patching",
        "--patch_layers",
        SIX_LAYER_PROFILE_LAYERS,
        "--csv_path",
        str(support_root / "gemma2b_raw_6layer_full_seed42.csv"),
    ]
    if local_files_only:
        raw.append("--local_files_only")

    clt = [
        sys.executable,
        "aom_eval.py",
        "--model_name_or_path",
        PAPER_MODEL_REPO_ID,
        "--revision",
        MODEL_REVISION,
        "--device",
        PAPER_SUPPORT_DEVICE,
        "--attn_implementation",
        "eager",
        "--disamb_path",
        "data/disamb_pairs.jsonl",
        "--cf_path",
        "data/counterfactual.jsonl",
        "--coh_path",
        "data/coherence.jsonl",
        "--bootstrap_n",
        "1000",
        "--bootstrap_seed",
        "42",
        "--ci",
        "0.95",
        "--seed",
        "42",
        "--run_clt_patching",
        "--clt_repo",
        str(ROOT / README_CORE_BUNDLE_PATH),
        "--clt_layers",
        SIX_LAYER_PROFILE_LAYERS,
        "--clt_width",
        "16k",
        "--clt_scale",
        "1.0",
        "--csv_path",
        str(support_root / "gemma2b_clt_6layer_full_seed42.csv"),
    ]
    if local_files_only:
        clt.append("--local_files_only")

    sae = [
        sys.executable,
        "aom_eval.py",
        "--model_name_or_path",
        PAPER_MODEL_REPO_ID,
        "--revision",
        MODEL_REVISION,
        "--device",
        PAPER_SUPPORT_DEVICE,
        "--torch_dtype",
        "float32",
        "--attn_implementation",
        "eager",
        "--disamb_path",
        "data_paper_hardened_v2/disamb_pairs.jsonl",
        "--cf_path",
        "data_paper_hardened_v2/counterfactual.jsonl",
        "--coh_path",
        "data_paper_hardened_v2/coherence.jsonl",
        "--dataset_manifest_path",
        "data_paper_hardened_v2/DATASET_MANIFEST.json",
        "--bootstrap_n",
        "1000",
        "--bootstrap_seed",
        "42",
        "--ci",
        "0.95",
        "--seed",
        "20260224",
        "--run_patching",
        "--run_patching_specificity",
        "--run_sae_patching",
        "--sae_repo",
        f"hf://{PAPER_SCOPE_REPO_ID}@{SAE_REVISION}",
        "--sae_width",
        "16k",
        "--sae_run_name",
        FIXED_LAYER_SAE_RUN_NAME,
        "--sae_layers",
        FIXED_LAYER_SAE_LAYER,
        "--sae_scale",
        "1.0",
        "--strict_finite",
        "--strict_metrics",
        "--protocol_path",
        "configs/mom_flagship_protocol.yaml",
        "--require_git",
        "--csv_path",
        str(support_root / "gemma2b_sae.csv"),
    ]
    if local_files_only:
        sae.append("--local_files_only")

    return [
        CommandSpec(name="Six-layer raw layer profile", argv=tuple(raw), artifact_kind="support"),
        CommandSpec(name="Six-layer CLT layer profile", argv=tuple(clt), artifact_kind="support"),
        CommandSpec(name="Fixed-layer SAE specificity support", argv=tuple(sae), artifact_kind="support"),
    ]


def _artifact_failed(kind: str, *, core_failed: bool, support_failed: bool) -> tuple[bool, bool]:
    if str(kind).lower() == "support":
        return core_failed, True
    if str(kind).lower() == "core":
        return True, support_failed
    return True, True


def _runtime_provenance() -> dict[str, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive only
        torch_version = f"unavailable ({exc.__class__.__name__})"
    else:
        torch_version = str(getattr(torch, "__version__", "unknown"))

    try:
        import transformers
    except Exception as exc:  # pragma: no cover - defensive only
        transformers_version = f"unavailable ({exc.__class__.__name__})"
    else:
        transformers_version = str(getattr(transformers, "__version__", "unknown"))

    return {
        "python_version": str(sys.version.split()[0]),
        "platform": str(platform.platform()),
        "torch_version": torch_version,
        "transformers_version": transformers_version,
    }


def _display_provenance_path(path: Path, run_root: Path) -> str:
    try:
        return f"$RUN_ROOT/{path.relative_to(run_root).as_posix()}"
    except ValueError:
        return _display_reference_path(path)


def _support_artifact_provenance_lines(run_root: Path) -> list[str]:
    lines: list[str] = []
    for label, rel in SUPPORT_MANIFEST_SPECS:
        try:
            loaded = _load_result_row(run_root / rel)
        except FileNotFoundError:
            continue
        except Exception as exc:
            lines.append(f"- {label}: unavailable=`{exc.__class__.__name__}: {exc}`")
            continue
        row = loaded.row
        lines.append(
            f"- {label}: row_source=`{loaded.source_kind}`; "
            f"row_path=`{_display_provenance_path(loaded.source_path, run_root)}`; "
            f"requested_device=`{row.get('requested_device')}`; "
            f"observed_device=`{row.get('device')}`; "
            f"model_param_dtype=`{row.get('model_param_dtype')}`; "
            f"logprobs_dtype=`{row.get('logprobs_dtype')}`; "
            f"seed=`{row.get('seed')}`; "
            f"bootstrap_seed=`{row.get('bootstrap_seed')}`"
        )
    return lines


def _status_summary(*, records: list[CommandRecord], missing, checks) -> tuple[str, str, str]:
    core_failed = False
    support_failed = False

    for record in records:
        if int(record.exit_code) == 0:
            continue
        core_failed, support_failed = _artifact_failed(record.artifact_kind, core_failed=core_failed, support_failed=support_failed)

    for artifact in missing:
        core_failed, support_failed = _artifact_failed(artifact.artifact_kind, core_failed=core_failed, support_failed=support_failed)

    for check in checks:
        if not _is_failure_status(check.status):
            continue
        core_failed, support_failed = _artifact_failed(check.artifact_kind, core_failed=core_failed, support_failed=support_failed)

    core_status = "FAIL" if core_failed else "PASS"
    support_status = "FAIL" if support_failed else "PASS"
    overall_status = "FAIL" if (core_failed or support_failed) else "PASS"
    return core_status, support_status, overall_status


def _render_report(
    *,
    run_root: Path,
    created_run_root: bool,
    records: list[CommandRecord],
    missing: list[Path],
    checks,
    dry_run: bool,
) -> str:
    if dry_run:
        core_status = "NOT_EXECUTED"
        support_status = "NOT_EXECUTED"
        overall_status = "PLAN_ONLY"
    else:
        core_status, support_status, overall_status = _status_summary(
            records=records,
            missing=missing,
            checks=checks,
        )
    provenance = _runtime_provenance()
    lines = [
        "# MoM Paper Reproduction Report",
        "",
        f"- run root: `$RUN_ROOT` (`{run_root.name}`)",
        f"- run root created by script: `{'yes' if created_run_root else 'no'}`",
        f"- repository commit: `{subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()}`",
        f"- mode: `{'dry_run' if dry_run else 'execute'}`",
    ]

    lines.extend(["", "## Command Log", ""])
    for index, record in enumerate(records, start=1):
        status = "NOT_EXECUTED" if dry_run else ("PASS" if record.exit_code == 0 else "FAIL")
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

    lines.extend(
        [
            "## Status Summary",
            "",
            f"- core_claims_status: `{core_status}`",
            f"- support_artifacts_status: `{support_status}`",
            f"- overall_status: `{overall_status}`",
            "",
        ]
    )

    lines.extend(
        [
            "## Provenance",
            "",
            f"- python_version: `{provenance['python_version']}`",
            f"- platform: `{provenance['platform']}`",
            f"- torch_version: `{provenance['torch_version']}`",
            f"- transformers_version: `{provenance['transformers_version']}`",
            "",
        ]
    )

    if not dry_run:
        support_lines = _support_artifact_provenance_lines(run_root)
        if support_lines:
            lines.extend(["## Support Artifact Provenance", "", *support_lines, ""])

    if dry_run:
        lines.extend(
            [
                "## Dry Run",
                "",
                "- status: `PLAN_ONLY`",
                "- note: `No commands were executed; this report lists the exact paper-facing command sequence.`",
            ]
        )
        return "\n".join(lines) + "\n"

    if missing:
        lines.extend(["## Missing Outputs", ""])
        for artifact in missing:
            path = artifact.path
            try:
                rel = path.relative_to(run_root).as_posix()
                lines.append(f"- `$RUN_ROOT/{rel}`")
            except ValueError:
                lines.append(f"- `{path}`")
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.extend(["## Output Files", ""])
    for path in sorted(path.relative_to(run_root).as_posix() for path in run_root.rglob("*") if path.is_file()):
        lines.append(f"- `$RUN_ROOT/{path}`")

    lines.extend(["", "## Numeric Checks", ""])
    for check in checks:
        lines.append(_render_check(check))

    lines.extend(["", "## Overall", "", f"- overall_status: `{overall_status}`"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the paper-facing MoM package: core CLT comparability/decomposition plus the SAE boundary support run."
    )
    parser.add_argument(
        "--run_root",
        default="",
        help="Optional run directory. Defaults to a fresh temp directory outside the repo.",
    )
    parser.add_argument(
        "--report_path",
        default="",
        help="Markdown report path. Defaults to `$RUN_ROOT/mom_paper_reproduction_report.md`.",
    )
    parser.add_argument(
        "--json_log_path",
        default="",
        help="Machine-readable command/check log path. Defaults to `$RUN_ROOT/mom_paper_reproduction_log.json`.",
    )
    parser.add_argument(
        "--skip_bundle_materialization",
        action="store_true",
        help="Do not materialize the CLT bundle even if it is missing.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Use only local HF cache for model / SAE / CLT artifact resolution.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print and log the exact commands without executing them.",
    )
    args = parser.parse_args(argv)

    created_run_root = False
    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
    else:
        run_root = Path(tempfile.mkdtemp(prefix="mom_paper_"))
        created_run_root = True

    if any(run_root.iterdir()):
        print(f"[error] Run root must start empty: {run_root}", file=sys.stderr)
        return 2

    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else run_root / "mom_paper_reproduction_report.md"
    json_log_path = Path(args.json_log_path).expanduser().resolve() if args.json_log_path else run_root / "mom_paper_reproduction_log.json"

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
            _write_json_log(json_log_path, records, [])
            return 2

    specs = [
        *_readme_command_specs(run_root, local_files_only=bool(args.local_files_only)),
        *_paper_support_command_specs(run_root, local_files_only=bool(args.local_files_only)),
    ]
    for spec in specs:
        record = _run_one(spec, cwd=ROOT, run_root=run_root, dry_run=bool(args.dry_run))
        records.append(record)
        if record.exit_code != 0:
            break

    if args.dry_run:
        missing = []
        checks = []
    else:
        missing, checks = verify_run(run_root)

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
    _write_json_log(json_log_path, records, checks)

    if args.dry_run:
        print(report)
        return 0
    if any(record.exit_code != 0 for record in records):
        print(report, file=sys.stderr)
        return 2
    if missing or any(_is_failure_status(check.status) for check in checks):
        print(report, file=sys.stderr)
        return 2

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
