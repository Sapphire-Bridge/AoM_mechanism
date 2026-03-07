#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_readme_reproduction import (
    README_CORE_BUNDLE_PATH,
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
from scripts.verify_mom_paper import verify_run


MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
SAE_REVISION = "fd571b47c1c64851e9b1989792367b9babb4af63"
SIX_LAYER_PROFILE_LAYERS = "4,8,12,16,20,24"
FIXED_LAYER_SAE_RUN_NAME = "average_l0_457"
FIXED_LAYER_SAE_LAYER = "24"


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
    )


def _paper_support_command_specs(run_root: Path, *, local_files_only: bool) -> list[CommandSpec]:
    support_root = run_root / "paper_support"

    raw = [
        sys.executable,
        "aom_eval.py",
        "--model_name_or_path",
        "google/gemma-2-2b",
        "--revision",
        MODEL_REVISION,
        "--device",
        "auto",
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
        "google/gemma-2-2b",
        "--revision",
        MODEL_REVISION,
        "--device",
        "auto",
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
        "google/gemma-2-2b",
        "--revision",
        MODEL_REVISION,
        "--device",
        "auto",
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
        f"hf://google/gemma-scope-2b-pt-res@{SAE_REVISION}",
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
        CommandSpec(name="Six-layer raw layer profile", argv=tuple(raw)),
        CommandSpec(name="Six-layer CLT layer profile", argv=tuple(clt)),
        CommandSpec(name="Fixed-layer SAE specificity support", argv=tuple(sae)),
    ]


def _render_report(
    *,
    run_root: Path,
    created_run_root: bool,
    records: list[CommandRecord],
    missing: list[Path],
    checks,
    dry_run: bool,
) -> str:
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

    if dry_run:
        lines.extend(
            [
                "## Dry Run",
                "",
                "- status: `PASS`",
                "- note: `No commands were executed; this report lists the exact paper-facing command sequence.`",
            ]
        )
        return "\n".join(lines) + "\n"

    if missing:
        lines.extend(["## Missing Outputs", ""])
        for path in missing:
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

    overall = "PASS"
    if any(record.exit_code != 0 for record in records) or any(check.status != "pass" for check in checks):
        overall = "FAIL"
    lines.extend(["", "## Overall", "", f"- status: `{overall}`"])
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
    if missing or any(check.status != "pass" for check in checks):
        print(report, file=sys.stderr)
        return 2

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
