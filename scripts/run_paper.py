from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.io import write_jsonl


M1MAX_MODELS: list[str] = [
    "gpt2",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
]

A100_EXTRA_MODELS: list[str] = [
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]

CF_PATCHING_MODELS: list[str] = [
    "gpt2",
    "Qwen/Qwen2.5-3B",
]

COH_PATCHING_MODELS: list[str] = [
    "gpt2",
]


@dataclass
class TimedCommandResult:
    returncode: int
    timed_out: bool


@dataclass
class BehavioralModelResult:
    model: str
    status: str
    csv_path: str
    reason: str = ""
    exit_code: int | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _shlex_join(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(a)) for a in argv)


def _display_arg(arg: str | Path) -> str:
    s = str(arg)
    if s == sys.executable:
        return "python"
    try:
        p = Path(s)
    except Exception:
        return s
    if p.is_absolute():
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            return s
        return rel.as_posix()
    return s


def _portable_manifest_path(path: str | Path, *, base_dir: str | Path) -> str:
    return Path(os.path.relpath(str(path), str(base_dir))).as_posix()


def _run(argv: List[str], *, cwd: Path, dry_run: bool) -> None:
    print(_shlex_join(_display_arg(arg) for arg in argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, cwd=str(cwd), check=True)


def _terminate_process_group(proc: subprocess.Popen[bytes], *, grace_seconds: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


def _run_with_timeout(
    argv: List[str],
    *,
    cwd: Path,
    dry_run: bool,
    timeout_seconds: int | None,
) -> TimedCommandResult:
    print(_shlex_join(_display_arg(arg) for arg in argv), flush=True)
    if dry_run:
        return TimedCommandResult(returncode=0, timed_out=False)
    proc = subprocess.Popen(argv, cwd=str(cwd), start_new_session=True)
    try:
        if timeout_seconds is None or int(timeout_seconds) <= 0:
            return TimedCommandResult(returncode=int(proc.wait()), timed_out=False)
        return TimedCommandResult(returncode=int(proc.wait(timeout=float(timeout_seconds))), timed_out=False)
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc)
        return TimedCommandResult(returncode=124, timed_out=True)
    except KeyboardInterrupt:
        _terminate_process_group(proc)
        raise


def _model_slug(model_name: str) -> str:
    return str(model_name).replace("/", "_")


def _merge_csv_files(csv_paths: List[Path], out_path: Path) -> None:
    if not csv_paths:
        return
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in csv_paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for name in reader.fieldnames:
                    if name not in fieldnames:
                        fieldnames.append(name)
            for row in reader:
                rows.append({str(k): "" if v is None else str(v) for k, v in row.items()})
    if not fieldnames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _relative_to_results_dir(path: str | Path, *, results_dir: Path) -> str:
    raw = str(path)
    if not raw:
        return ""
    p = Path(raw)
    try:
        return p.relative_to(results_dir).as_posix()
    except ValueError:
        return raw


def _render_behavioral_status_markdown(results: List[BehavioralModelResult], *, results_dir: Path) -> str:
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    timed_out = sum(1 for r in results if r.status == "TIMEOUT")
    overall = "PASS" if failed == 0 and timed_out == 0 else "FAIL"
    lines = [
        "## MPS Behavioral Model Status",
        "",
        f"- overall_status: {overall}",
        f"- passed_models: {passed}",
        f"- failed_models: {failed}",
        f"- timed_out_models: {timed_out}",
        "",
        "| model | status | csv_path | detail |",
        "|---|---|---|---|",
    ]
    for r in results:
        rel_csv = _relative_to_results_dir(r.csv_path, results_dir=results_dir)
        detail = r.reason or ("" if r.exit_code is None else f"exit {int(r.exit_code)}")
        lines.append(f"| {r.model} | {r.status} | {rel_csv} | {detail} |")
    lines.append("")
    return "\n".join(lines)


def _write_behavioral_status_artifacts(results_dir: Path, results: List[BehavioralModelResult]) -> tuple[Path, Path]:
    md_text = _render_behavioral_status_markdown(results, results_dir=results_dir)
    payload = {
        "generated_at_utc": _utc_now_iso(),
        "overall_status": "PASS" if all(r.status == "PASS" for r in results) else "FAIL",
        "models": [
            {
                "model": str(r.model),
                "status": str(r.status),
                "csv_path": _relative_to_results_dir(r.csv_path, results_dir=results_dir),
                "reason": str(r.reason),
                "exit_code": None if r.exit_code is None else int(r.exit_code),
            }
            for r in results
        ],
    }
    md_path = results_dir / "behavioral_model_status.md"
    json_path = results_dir / "behavioral_model_status.json"
    md_path.write_text(md_text + "\n", encoding="utf-8")
    _write_json(json_path, payload)
    return md_path, json_path


def _append_markdown_to_report(report_path: Path, markdown_text: str) -> None:
    if not report_path.exists():
        return
    existing = report_path.read_text(encoding="utf-8")
    suffix = markdown_text.strip()
    if suffix and suffix not in existing:
        report_path.write_text(existing.rstrip() + "\n\n" + suffix + "\n", encoding="utf-8")


def _try_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return ""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_nonempty_lines(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _parse_int_csv(raw: str) -> List[int]:
    vals: List[int] = []
    for part in str(raw or "").replace(";", ",").split(","):
        s = str(part).strip()
        if not s:
            continue
        vals.append(int(s))
    return vals


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_paper_dataset(
    *,
    data_dir: Path,
    seed: int,
    disamb_mode: str = "hardened",
    cf_include_shams: bool = True,
    cf_include_graded: bool = True,
    n_coh: int = 80,
    coh_include_controls: bool = True,
    dry_run: bool,
) -> Path:
    """
    Ensure a paper-canonical dataset exists, and write a deterministic manifest.

    This is intentionally separate from the repo's checked-in `data/` so you can pin a "paper dataset"
    without overwriting examples.
    """
    disamb_path = data_dir / "disamb_pairs.jsonl"
    cf_path = data_dir / "counterfactual.jsonl"
    coh_path = data_dir / "coherence.jsonl"

    files_exist = bool(disamb_path.exists() and cf_path.exists() and coh_path.exists())

    if not files_exist:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "generate_data.py"),
            "--out_dir",
            str(data_dir),
            "--seed",
            str(int(seed)),
            "--disamb_mode",
            str(disamb_mode),
            "--n_coh",
            str(int(n_coh)),
        ]
        if cf_include_shams:
            cmd.append("--cf_include_shams")
        if cf_include_graded:
            cmd.append("--cf_include_graded")
        if coh_include_controls:
            cmd.append("--coh_include_controls")
        _run(cmd, cwd=ROOT, dry_run=dry_run)

    manifest_path = data_dir / "DATASET_MANIFEST.json"
    files_exist = bool(disamb_path.exists() and cf_path.exists() and coh_path.exists())
    if files_exist:
        from collections import Counter

        from aom.data.bundle_manifest import compute_bundle_id

        def _paper_bundle_name() -> str:
            cfg = (
                str(disamb_mode),
                bool(cf_include_shams),
                bool(cf_include_graded),
                int(n_coh),
                bool(coh_include_controls),
            )
            if cfg == ("hardened", True, False, 40, True):
                return "paper_hardened_v1"
            if cfg == ("hardened", True, True, 80, True):
                return "paper_hardened_v2"
            if str(disamb_mode) == "hardened":
                return "paper_hardened_custom"
            return "paper_dataset"

        def _iter_jsonl(path: Path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)

        files = {
            "disamb_pairs.jsonl": {
                "path": _portable_manifest_path(disamb_path, base_dir=data_dir.parent),
                "n_lines": int(_count_nonempty_lines(disamb_path)),
                "sha256": _sha256_file(disamb_path),
            },
            "counterfactual.jsonl": {
                "path": _portable_manifest_path(cf_path, base_dir=data_dir.parent),
                "n_lines": int(_count_nonempty_lines(cf_path)),
                "sha256": _sha256_file(cf_path),
            },
            "coherence.jsonl": {
                "path": _portable_manifest_path(coh_path, base_dir=data_dir.parent),
                "n_lines": int(_count_nonempty_lines(coh_path)),
                "sha256": _sha256_file(coh_path),
            },
        }
        bundle_id = compute_bundle_id({"files": files})

        dis_pair_variant = Counter()
        for row in _iter_jsonl(disamb_path):
            meta = row.get("metadata", None)
            if isinstance(meta, dict):
                dis_pair_variant[str(meta.get("pair_variant", ""))] += 1
        cf_expected_effect = Counter()
        for row in _iter_jsonl(cf_path):
            cf_expected_effect[str(row.get("expected_effect", "shift"))] += 1
        coh_group = Counter()
        coh_constraint_type = Counter()
        coh_n_constraints = Counter()
        coh_constraint_type_main = Counter()
        coh_n_constraints_main = Counter()
        for row in _iter_jsonl(coh_path):
            grp = str(row.get("group", "main"))
            coh_group[grp] += 1
            coh_constraint_type[str(row.get("constraint_type", ""))] += 1
            meta = row.get("metadata", None)
            nc = 1
            if isinstance(meta, dict):
                try:
                    nc = int(meta.get("n_constraints", 1))
                except Exception:
                    nc = 1
            coh_n_constraints[str(int(nc))] += 1
            if grp == "main":
                coh_constraint_type_main[str(row.get("constraint_type", ""))] += 1
                coh_n_constraints_main[str(int(nc))] += 1

        suite_counts = {
            "disamb": {
                "n_pairs_total": int(files["disamb_pairs.jsonl"]["n_lines"]),
                "pair_variant_counts": {str(k): int(v) for k, v in sorted(dis_pair_variant.items())},
            },
            "cf": {
                "n_items_total": int(files["counterfactual.jsonl"]["n_lines"]),
                "expected_effect_counts": {str(k): int(v) for k, v in sorted(cf_expected_effect.items())},
            },
            "coh": {
                "n_rows_total": int(files["coherence.jsonl"]["n_lines"]),
                "group_counts": {str(k): int(v) for k, v in sorted(coh_group.items())},
                "constraint_type_counts": {str(k): int(v) for k, v in sorted(coh_constraint_type.items())},
                "n_constraints_counts": {str(k): int(v) for k, v in sorted(coh_n_constraints.items())},
                "main": {
                    "n_items": int(coh_group.get("main", 0)),
                    "constraint_type_counts": {str(k): int(v) for k, v in sorted(coh_constraint_type_main.items())},
                    "n_constraints_counts": {str(k): int(v) for k, v in sorted(coh_n_constraints_main.items())},
                },
            },
        }

        expected_cfg = {
            "name": _paper_bundle_name(),
            "seed": int(seed),
            "disamb_mode": str(disamb_mode),
            "cf_include_shams": bool(cf_include_shams),
            "cf_include_graded": bool(cf_include_graded),
            "n_coh": int(n_coh),
            "coh_include_controls": bool(coh_include_controls),
        }

        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                raise ValueError(f"Existing DATASET_MANIFEST.json is not a JSON object: {str(manifest_path)}")
            mismatches = []
            for k, v in expected_cfg.items():
                if k in existing and existing.get(k) != v:
                    mismatches.append(f"{k}: have={existing.get(k)!r} want={v!r}")
            for filename, info in files.items():
                have = existing.get("files", {}).get(filename, {}).get("sha256", None)
                if have is not None and str(have).strip() != str(info["sha256"]).strip():
                    mismatches.append(f"files.{filename}.sha256: have={str(have).strip()} want={str(info['sha256']).strip()}")
            if mismatches:
                raise ValueError(
                    "Dataset directory already contains a different dataset/manifest configuration. "
                    f"Use a fresh --data_dir or delete the existing one. Mismatches: {mismatches}"
                )

            updated = dict(existing)
            for k, v in expected_cfg.items():
                updated.setdefault(k, v)
            updated.setdefault("generator", "scripts/generate_data.py")
            updated.setdefault("git_commit", _try_git_commit())
            updated.setdefault("generated_at_utc", _utc_now_iso())
            updated["files"] = files
            updated["bundle_id"] = str(bundle_id)
            updated["suite_counts"] = suite_counts
            if not dry_run and updated != existing:
                _write_json(manifest_path, updated)
        else:
            manifest = {
                **expected_cfg,
                "generated_at_utc": _utc_now_iso(),
                "git_commit": _try_git_commit(),
                "generator": "scripts/generate_data.py",
                "files": files,
                "bundle_id": str(bundle_id),
                "suite_counts": suite_counts,
            }
            if not dry_run:
                _write_json(manifest_path, manifest)

    return manifest_path


def _make_wordlevel_tokenizer_dir(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "[PAD]": 0,
        "[EOS]": 1,
        "[UNK]": 2,
        "Alice": 3,
        "went": 4,
        "to": 5,
        "the": 6,
        "bank": 7,
        "and": 8,
        "then": 9,
        "sat": 10,
        "by": 11,
        "river": 12,
        "loan": 13,
        "did": 14,
        "not": 15,
        "go": 16,
        "store": 17,
        "Therefore": 18,
        "John": 19,
        "died": 20,
        "Later": 21,
        "was": 22,
        "remembered": 23,
        "walked": 24,
        "quietly": 25,
        "today": 26,
    }

    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(path)


def _make_tiny_local_gpt2_model_dir(path: Path) -> None:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=32,
        vocab_size=64,
        n_positions=64,
        pad_token_id=0,
        eos_token_id=1,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    model.save_pretrained(path)


def ensure_smoke_local_model(model_dir: Path) -> None:
    """
    Create a tiny local model+tokenizer so `aom_eval.py` can run fully offline.
    """
    if (model_dir / "config.json").exists() and (model_dir / "tokenizer.json").exists():
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    _make_wordlevel_tokenizer_dir(model_dir)
    _make_tiny_local_gpt2_model_dir(model_dir)


def ensure_smoke_datasets(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    disamb_path = data_dir / "disamb_pairs.jsonl"
    cf_path = data_dir / "counterfactual.jsonl"
    coh_path = data_dir / "coherence.jsonl"

    if not disamb_path.exists():
        write_jsonl(
            [
                {
                    "pair_id": "bank-smoke-0",
                    "target": "bank",
                    "target_occurrence": 0,
                    "a": {"prompt": "Alice went to the bank and then Alice went to the store Therefore Alice", "expected_label": "loan"},
                    "b": {"prompt": "Alice sat by the bank and then Alice sat by the river Later Alice", "expected_label": "river"},
                    "choices": {"loan": [" loan"], "river": [" river"]},
                    "metadata": {"type": "smoke"},
                }
            ],
            disamb_path,
        )

    if not cf_path.exists():
        write_jsonl(
            [
                {
                    "item_id": "negation-smoke-0",
                    "base": {"prompt": "Alice did go to the store Therefore Alice", "expected_label": "did"},
                    "cf": {"prompt": "Alice did not go to the store Therefore Alice", "expected_label": "not"},
                    "choices": {"did": [" did"], "not": [" not"]},
                    "intervention_type": "negation",
                    "contrast_labels": ["did", "not"],
                    "expected_effect": "shift",
                    "metadata": {"type": "smoke"},
                },
                {
                    "item_id": "negation-smoke-0__sham",
                    "base": {"prompt": "Alice did go to the store Therefore Alice", "expected_label": "did"},
                    "cf": {"prompt": "Alice did go to the store Therefore, Alice", "expected_label": "did"},
                    "choices": {"did": [" did"], "not": [" not"]},
                    "intervention_type": "sham_punctuation",
                    "contrast_labels": ["did", "not"],
                    "expected_effect": "invariant",
                    "metadata": {"type": "smoke", "control": "sham"},
                },
            ],
            cf_path,
        )

    if not coh_path.exists():
        write_jsonl(
            [
                {
                    "item_id": "entity-smoke-0__main",
                    "context": "John died Later John",
                    "valid_continuations": [" was remembered"],
                    "invalid_continuations": [" walked"],
                    "constraint_type": "entity_state",
                    "group": "main",
                    "metadata": {"type": "smoke"},
                }
            ],
            coh_path,
        )

    return {"disamb": disamb_path, "cf": cf_path, "coh": coh_path}


@dataclass(frozen=True)
class PaperRun:
    mode: str
    results_dir: Path
    dataset_manifest_path: Optional[Path]
    commands: List[List[str]]


def _write_run_manifest(run: PaperRun) -> None:
    out = {
        "mode": str(run.mode),
        "generated_at_utc": _utc_now_iso(),
        "git_commit": _try_git_commit(),
        "results_dir": str(run.results_dir),
        "dataset_manifest_path": "" if run.dataset_manifest_path is None else str(run.dataset_manifest_path),
        "commands": [list(cmd) for cmd in run.commands],
    }
    _write_json(run.results_dir / "RUN_MANIFEST.json", out)


def _aom_eval_cmd(
    *,
    models: List[str],
    device: str,
    torch_dtype: Optional[str],
    attn_implementation: str,
    local_files_only: bool,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    disamb_path: str,
    cf_path: str,
    coh_path: str,
    dataset_manifest_path: Optional[str] = None,
    bootstrap_n: int,
    bootstrap_seed: int,
    ci: float,
    csv_path: str,
    run_patching: bool = False,
    patch_layers: str = "",
    run_patching_specificity: bool = False,
    patch_specificity_depth_frac: Optional[float] = None,
    patch_specificity_layer: Optional[int] = None,
    patch_specificity_buffer: Optional[int] = None,
    patch_specificity_position_window: Optional[int] = None,
    patch_specificity_seed: Optional[int] = None,
    device_map: Optional[str] = None,
    run_clt_patching: bool = False,
    clt_repo: str = "",
    clt_width: str = "16k",
    clt_run_name: Optional[str] = None,
    clt_l0_target: Optional[int] = None,
    clt_layers: str = "",
    clt_scale: float = 1.0,
    clt_dtype: str = "float32",
    clt_decode_strategy: str = "delta_1decode",
    clt_dtype_policy: str = "clt",
    clt_eps_active: float = 1e-6,
) -> List[str]:
    argv = [sys.executable, str(ROOT / "aom_eval.py")]
    argv += ["--models", *models]
    argv += ["--device", str(device)]
    argv += ["--attn_implementation", str(attn_implementation)]
    if torch_dtype is not None:
        argv += ["--torch_dtype", str(torch_dtype)]
    if device_map is not None:
        argv += ["--device_map", str(device_map)]
    if local_files_only:
        argv.append("--local_files_only")
    if trust_remote_code:
        argv.append("--trust_remote_code")
    if revision is not None and str(revision).strip():
        argv += ["--revision", str(revision)]
    if tokenizer_revision is not None and str(tokenizer_revision).strip():
        argv += ["--tokenizer_revision", str(tokenizer_revision)]
    argv += ["--disamb_path", str(disamb_path)]
    argv += ["--cf_path", str(cf_path)]
    argv += ["--coh_path", str(coh_path)]
    if dataset_manifest_path is not None and str(dataset_manifest_path).strip():
        argv += ["--dataset_manifest_path", str(dataset_manifest_path)]
    argv += ["--bootstrap_n", str(int(bootstrap_n))]
    argv += ["--bootstrap_seed", str(int(bootstrap_seed))]
    argv += ["--ci", str(float(ci))]
    if run_patching:
        argv.append("--run_patching")
        if str(patch_layers).strip():
            argv += ["--patch_layers", str(patch_layers)]
    if run_patching_specificity:
        argv.append("--run_patching_specificity")
        if patch_specificity_layer is not None:
            argv += ["--patch_specificity_layer", str(int(patch_specificity_layer))]
        if patch_specificity_depth_frac is not None:
            argv += ["--patch_specificity_depth_frac", str(float(patch_specificity_depth_frac))]
        if patch_specificity_buffer is not None:
            argv += ["--patch_specificity_buffer", str(int(patch_specificity_buffer))]
        if patch_specificity_position_window is not None:
            argv += ["--patch_specificity_position_window", str(int(patch_specificity_position_window))]
        if patch_specificity_seed is not None:
            argv += ["--patch_specificity_seed", str(int(patch_specificity_seed))]
    if run_clt_patching:
        if not str(clt_repo).strip():
            raise ValueError("run_clt_patching requires clt_repo")
        if not str(clt_layers).strip():
            raise ValueError("run_clt_patching requires clt_layers")
        argv.append("--run_clt_patching")
        argv += ["--clt_repo", str(clt_repo)]
        argv += ["--clt_width", str(clt_width)]
        if clt_run_name is not None and str(clt_run_name).strip():
            argv += ["--clt_run_name", str(clt_run_name)]
        if clt_l0_target is not None:
            argv += ["--clt_l0_target", str(int(clt_l0_target))]
        argv += ["--clt_layers", str(clt_layers)]
        argv += ["--clt_scale", str(float(clt_scale))]
        argv += ["--clt_dtype", str(clt_dtype)]
        argv += ["--clt_decode_strategy", str(clt_decode_strategy)]
        argv += ["--clt_dtype_policy", str(clt_dtype_policy)]
        argv += ["--clt_eps_active", str(float(clt_eps_active))]
    argv += ["--csv_path", str(csv_path)]
    return argv


def _cf_patching_cmd(
    *,
    models: List[str],
    device: str,
    torch_dtype: Optional[str],
    local_files_only: bool,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    cf_path: str,
    dataset_manifest_path: Optional[str] = None,
    bootstrap_n: int,
    bootstrap_seed: int,
    ci: float,
    csv_path: str,
    patch_layers: str = "",
) -> List[str]:
    argv = [sys.executable, str(ROOT / "aom_cf_patching.py")]
    argv += ["--models", *models]
    argv += ["--device", str(device)]
    argv += ["--attn_implementation", "eager"]
    if torch_dtype is not None:
        argv += ["--torch_dtype", str(torch_dtype)]
    if local_files_only:
        argv.append("--local_files_only")
    if trust_remote_code:
        argv.append("--trust_remote_code")
    if revision is not None and str(revision).strip():
        argv += ["--revision", str(revision)]
    if tokenizer_revision is not None and str(tokenizer_revision).strip():
        argv += ["--tokenizer_revision", str(tokenizer_revision)]
    argv += ["--cf_path", str(cf_path)]
    if dataset_manifest_path is not None and str(dataset_manifest_path).strip():
        argv += ["--dataset_manifest_path", str(dataset_manifest_path)]
    argv += ["--bootstrap_n", str(int(bootstrap_n))]
    argv += ["--bootstrap_seed", str(int(bootstrap_seed))]
    argv += ["--ci", str(float(ci))]
    if str(patch_layers).strip():
        argv += ["--patch_layers", str(patch_layers)]
    argv += ["--csv_path", str(csv_path)]
    return argv


def _coh_patching_cmd(
    *,
    models: List[str],
    device: str,
    torch_dtype: Optional[str],
    local_files_only: bool,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    coh_path: str,
    dataset_manifest_path: Optional[str] = None,
    bootstrap_n: int,
    bootstrap_seed: int,
    ci: float,
    csv_path: str,
    patch_layers: str = "",
) -> List[str]:
    argv = [sys.executable, str(ROOT / "aom_coh_patching.py")]
    argv += ["--models", *models]
    argv += ["--device", str(device)]
    argv += ["--attn_implementation", "eager"]
    if torch_dtype is not None:
        argv += ["--torch_dtype", str(torch_dtype)]
    if local_files_only:
        argv.append("--local_files_only")
    if trust_remote_code:
        argv.append("--trust_remote_code")
    if revision is not None and str(revision).strip():
        argv += ["--revision", str(revision)]
    if tokenizer_revision is not None and str(tokenizer_revision).strip():
        argv += ["--tokenizer_revision", str(tokenizer_revision)]
    argv += ["--coh_path", str(coh_path)]
    if dataset_manifest_path is not None and str(dataset_manifest_path).strip():
        argv += ["--dataset_manifest_path", str(dataset_manifest_path)]
    argv += ["--bootstrap_n", str(int(bootstrap_n))]
    argv += ["--bootstrap_seed", str(int(bootstrap_seed))]
    argv += ["--ci", str(float(ci))]
    if str(patch_layers).strip():
        argv += ["--patch_layers", str(patch_layers)]
    argv += ["--csv_path", str(csv_path)]
    return argv


def _disamb_path_decomp_cmd(
    *,
    model_name_or_path: str,
    device: str,
    torch_dtype: Optional[str],
    local_files_only: bool,
    trust_remote_code: bool,
    disamb_path: str,
    bootstrap_n: int,
    bootstrap_seed: int,
    ci: float,
    max_pairs: int,
    mode: str,
    position: int,
    rows_csv_path: str,
    summary_csv_path: str,
) -> List[str]:
    argv = [sys.executable, str(ROOT / "aom_disamb_path_decomp.py")]
    argv += ["--model_name_or_path", str(model_name_or_path)]
    argv += ["--device", str(device)]
    if torch_dtype is not None:
        argv += ["--torch_dtype", str(torch_dtype)]
    if local_files_only:
        argv.append("--local_files_only")
    if trust_remote_code:
        argv.append("--trust_remote_code")
    argv += ["--disamb_path", str(disamb_path)]
    argv += ["--bootstrap_n", str(int(bootstrap_n))]
    argv += ["--bootstrap_seed", str(int(bootstrap_seed))]
    argv += ["--ci", str(float(ci))]
    argv += ["--max_pairs", str(int(max_pairs))]
    argv += ["--mode", str(mode)]
    argv += ["--position", str(int(position))]
    argv += ["--rows_csv_path", str(rows_csv_path)]
    argv += ["--summary_csv_path", str(summary_csv_path)]
    return argv


def _why_fetch_cmd(
    *,
    task: str,
    model_name_or_path: str,
    device: str,
    torch_dtype: Optional[str],
    local_files_only: bool,
    trust_remote_code: bool,
    disamb_path: str,
    cf_path: str,
    coh_path: str,
    n_examples: int,
    heads_topk: int,
    bootstrap_n: int,
    bootstrap_seed: int,
    ci: float,
    rows_csv_path: str,
    summary_csv_path: str,
    smoke: bool,
) -> List[str]:
    argv = [sys.executable, str(ROOT / "aom_why_fetch.py")]
    argv += ["--task", str(task)]
    argv += ["--model_name_or_path", str(model_name_or_path)]
    argv += ["--device", str(device)]
    if torch_dtype is not None:
        argv += ["--torch_dtype", str(torch_dtype)]
    if local_files_only:
        argv.append("--local_files_only")
    if trust_remote_code:
        argv.append("--trust_remote_code")
    argv += ["--disamb_path", str(disamb_path)]
    argv += ["--cf_path", str(cf_path)]
    argv += ["--coh_path", str(coh_path)]
    argv += ["--n_examples", str(int(n_examples))]
    argv += ["--heads_topk", str(int(heads_topk))]
    argv += ["--bootstrap_n", str(int(bootstrap_n))]
    argv += ["--bootstrap_seed", str(int(bootstrap_seed))]
    argv += ["--ci", str(float(ci))]
    argv += ["--rows_csv_path", str(rows_csv_path)]
    argv += ["--summary_csv_path", str(summary_csv_path)]
    if bool(smoke):
        argv.append("--smoke")
    return argv


def _feature_families_cmd(
    *,
    model_name_or_path: str,
    sae_repo: str,
    sae_layer: int,
    sae_width: str,
    sae_scale: float,
    n_features: int,
    max_pairs: int,
    similarity_threshold: float,
    null_n: int,
    disamb_path: str,
    device: str,
    torch_dtype: Optional[str],
    local_files_only: bool,
    trust_remote_code: bool,
    bootstrap_seed: int,
    ci: float,
    out_prefix: str,
    smoke: bool,
) -> List[str]:
    argv = [sys.executable, str(ROOT / "aom_feature_families.py")]
    argv += ["--model_name_or_path", str(model_name_or_path)]
    argv += ["--sae_repo", str(sae_repo)]
    argv += ["--layer", str(int(sae_layer))]
    argv += ["--width", str(sae_width)]
    argv += ["--scale", str(float(sae_scale))]
    argv += ["--n_features", str(int(n_features))]
    argv += ["--max_pairs", str(int(max_pairs))]
    argv += ["--similarity_threshold", str(float(similarity_threshold))]
    argv += ["--null_n", str(int(null_n))]
    argv += ["--bootstrap_seed", str(int(bootstrap_seed))]
    argv += ["--ci", str(float(ci))]
    argv += ["--disamb_path", str(disamb_path)]
    argv += ["--device", str(device)]
    if torch_dtype is not None:
        argv += ["--torch_dtype", str(torch_dtype)]
    if local_files_only:
        argv.append("--local_files_only")
    if trust_remote_code:
        argv.append("--trust_remote_code")
    argv += ["--out_effect_matrix_csv", f"{out_prefix}_feature_effects.csv"]
    argv += ["--out_families_json", f"{out_prefix}_feature_families.json"]
    argv += ["--out_families_summary_csv", f"{out_prefix}_feature_families_summary.csv"]
    argv += ["--out_family_validation_csv", f"{out_prefix}_feature_family_validation.csv"]
    if bool(smoke):
        argv.append("--smoke")
    return argv


def _clt_topk_recovery_cmd(
    *,
    model_name_or_path: str,
    clt_repo: str,
    clt_width: str,
    clt_run_name: Optional[str],
    clt_l0_target: Optional[int],
    layers: str,
    ks: str,
    logz_ks: str,
    split_seed: int,
    frac_selection: float,
    random_k_seeds: str,
    random_control_mode: str,
    matched_bin_n_bins: int,
    eps: float,
    bootstrap_B: int,
    ci: float,
    with_logz: bool,
    disamb_path: str,
    device: str,
    torch_dtype: Optional[str],
    local_files_only: bool,
    trust_remote_code: bool,
    seed: int,
    out_csv: str,
    out_summary: str,
) -> List[str]:
    argv = [sys.executable, str(ROOT / "aom_clt_topk_recovery.py")]
    argv += ["--model_name_or_path", str(model_name_or_path)]
    argv += ["--clt_repo", str(clt_repo)]
    argv += ["--clt_width", str(clt_width)]
    if clt_run_name is not None and str(clt_run_name).strip():
        argv += ["--clt_run_name", str(clt_run_name)]
    if clt_l0_target is not None:
        argv += ["--clt_l0_target", str(int(clt_l0_target))]
    argv += ["--layers", str(layers)]
    argv += ["--ks", str(ks)]
    argv += ["--logz_ks", str(logz_ks)]
    argv += ["--split_seed", str(int(split_seed))]
    argv += ["--frac_selection", str(float(frac_selection))]
    argv += ["--random_k_seeds", str(random_k_seeds)]
    argv += ["--random_control_mode", str(random_control_mode)]
    argv += ["--matched_bin_n_bins", str(int(matched_bin_n_bins))]
    argv += ["--eps", str(float(eps))]
    argv += ["--bootstrap_B", str(int(bootstrap_B))]
    argv += ["--ci", str(float(ci))]
    argv += ["--disamb_path", str(disamb_path)]
    argv += ["--device", str(device)]
    if torch_dtype is not None:
        argv += ["--torch_dtype", str(torch_dtype)]
    if local_files_only:
        argv.append("--local_files_only")
    if trust_remote_code:
        argv.append("--trust_remote_code")
    if bool(with_logz):
        argv.append("--with_logz")
    argv += ["--seed", str(int(seed))]
    argv += ["--out_csv", str(out_csv)]
    argv += ["--out_summary", str(out_summary)]
    argv.append("--overwrite")
    return argv


def _clt_feature_analysis_cmd(
    *,
    model_name_or_path: str,
    clt_repo: str,
    clt_layer: int,
    clt_width: str,
    clt_run_name: Optional[str],
    clt_l0_target: Optional[int],
    topk_summary_path: str,
    top_n_features: int,
    max_pairs: int,
    disamb_path: str,
    device: str,
    torch_dtype: Optional[str],
    local_files_only: bool,
    trust_remote_code: bool,
    similarity_threshold: float,
    stability_bootstrap_n: int,
    null_n: int,
    bootstrap_seed: int,
    ci: float,
    out_prefix: str,
    smoke: bool,
) -> List[str]:
    argv = [sys.executable, str(ROOT / "aom_clt_feature_analysis.py")]
    argv += ["--model_name_or_path", str(model_name_or_path)]
    argv += ["--clt_repo", str(clt_repo)]
    argv += ["--clt_layer", str(int(clt_layer))]
    argv += ["--clt_width", str(clt_width)]
    if clt_run_name is not None and str(clt_run_name).strip():
        argv += ["--clt_run_name", str(clt_run_name)]
    if clt_l0_target is not None:
        argv += ["--clt_l0_target", str(int(clt_l0_target))]
    argv += ["--topk_summary_path", str(topk_summary_path)]
    argv += ["--top_n_features", str(int(top_n_features))]
    argv += ["--max_pairs", str(int(max_pairs))]
    argv += ["--disamb_path", str(disamb_path)]
    argv += ["--device", str(device)]
    if torch_dtype is not None:
        argv += ["--torch_dtype", str(torch_dtype)]
    if local_files_only:
        argv.append("--local_files_only")
    if trust_remote_code:
        argv.append("--trust_remote_code")
    argv += ["--similarity_threshold", str(float(similarity_threshold))]
    argv += ["--stability_bootstrap_n", str(int(stability_bootstrap_n))]
    argv += ["--null_n", str(int(null_n))]
    argv += ["--bootstrap_seed", str(int(bootstrap_seed))]
    argv += ["--ci", str(float(ci))]
    argv += ["--out_activation_csv", f"{out_prefix}_activations.csv"]
    argv += ["--out_top_tokens_csv", f"{out_prefix}_top_tokens.csv"]
    argv += ["--out_selectivity_csv", f"{out_prefix}_selectivity.csv"]
    argv += ["--out_families_json", f"{out_prefix}_families.json"]
    argv += ["--out_families_summary_csv", f"{out_prefix}_families_summary.csv"]
    argv += ["--out_family_validation_csv", f"{out_prefix}_families_validation.csv"]
    if bool(smoke):
        argv.append("--smoke")
    return argv


def _clt_head_attribution_cmd(
    *,
    model_name_or_path: str,
    clt_repo: str,
    clt_layer: int,
    clt_width: str,
    clt_run_name: Optional[str],
    clt_l0_target: Optional[int],
    topk_summary_path: str,
    top_n_features: int,
    head_layers: str,
    top_h: int,
    split_seed: int,
    frac_selection: float,
    disamb_path: str,
    device: str,
    torch_dtype: Optional[str],
    local_files_only: bool,
    trust_remote_code: bool,
    seed: int,
    max_pairs: int,
    ablate_positions: str,
    qk_patterns: bool,
    bootstrap_n: int,
    bootstrap_seed: int,
    ci: float,
    out_prefix: str,
    smoke: bool,
) -> List[str]:
    argv = [sys.executable, str(ROOT / "aom_clt_head_attribution.py")]
    argv += ["--model_name_or_path", str(model_name_or_path)]
    argv += ["--disamb_path", str(disamb_path)]
    argv += ["--clt_repo", str(clt_repo)]
    argv += ["--clt_layer", str(int(clt_layer))]
    argv += ["--clt_width", str(clt_width)]
    if clt_run_name is not None and str(clt_run_name).strip():
        argv += ["--clt_run_name", str(clt_run_name)]
    if clt_l0_target is not None:
        argv += ["--clt_l0_target", str(int(clt_l0_target))]
    argv += ["--topk_summary_path", str(topk_summary_path)]
    argv += ["--top_n_features", str(int(top_n_features))]
    argv += ["--head_layers", str(head_layers)]
    argv += ["--top_h", str(int(top_h))]
    argv += ["--split_seed", str(int(split_seed))]
    argv += ["--frac_selection", str(float(frac_selection))]
    argv += ["--seed", str(int(seed))]
    argv += ["--max_pairs", str(int(max_pairs))]
    argv += ["--ablate_positions", str(ablate_positions)]
    argv += ["--bootstrap_n", str(int(bootstrap_n))]
    argv += ["--bootstrap_seed", str(int(bootstrap_seed))]
    argv += ["--ci", str(float(ci))]
    argv += ["--device", str(device)]
    if torch_dtype is not None:
        argv += ["--torch_dtype", str(torch_dtype)]
    if local_files_only:
        argv.append("--local_files_only")
    if trust_remote_code:
        argv.append("--trust_remote_code")
    if bool(qk_patterns):
        argv.append("--qk_patterns")
    argv += ["--out_head_scores_csv", f"{out_prefix}_head_scores.csv"]
    argv += ["--out_ablation_rows_csv", f"{out_prefix}_ablation_rows.csv"]
    argv += ["--out_ablation_summary_csv", f"{out_prefix}_ablation_summary.csv"]
    argv += ["--out_qk_csv", f"{out_prefix}_qk_patterns.csv"]
    if bool(smoke):
        argv.append("--smoke")
    return argv


def _completeness_cmd(
    *,
    model_name_or_path: str,
    tasks: str,
    disamb_path: str,
    cf_path: str,
    coh_path: str,
    layers: str,
    explanation_path: str,
    device: str,
    torch_dtype: Optional[str],
    attn_implementation: str,
    local_files_only: bool,
    trust_remote_code: bool,
    bootstrap_n: int,
    bootstrap_seed: int,
    ci: float,
    csv_path: str,
    smoke: bool,
) -> List[str]:
    argv = [sys.executable, str(ROOT / "aom_completeness.py")]
    argv += ["--model_name_or_path", str(model_name_or_path)]
    argv += ["--tasks", str(tasks)]
    argv += ["--disamb_path", str(disamb_path)]
    argv += ["--cf_path", str(cf_path)]
    argv += ["--coh_path", str(coh_path)]
    argv += ["--device", str(device)]
    argv += ["--attn_implementation", str(attn_implementation)]
    if torch_dtype is not None:
        argv += ["--torch_dtype", str(torch_dtype)]
    if local_files_only:
        argv.append("--local_files_only")
    if trust_remote_code:
        argv.append("--trust_remote_code")
    if str(layers).strip():
        argv += ["--layers", str(layers)]
    if str(explanation_path).strip():
        argv += ["--explanation_path", str(explanation_path)]
    argv += ["--bootstrap_n", str(int(bootstrap_n))]
    argv += ["--bootstrap_seed", str(int(bootstrap_seed))]
    argv += ["--ci", str(float(ci))]
    argv += ["--csv_path", str(csv_path)]
    if bool(smoke):
        argv.append("--smoke")
    return argv


def run_smoke(args: argparse.Namespace) -> PaperRun:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest_path = None
    if not bool(args.skip_dataset):
        dataset_manifest_path = ensure_paper_dataset(
            data_dir=Path(args.data_dir),
            seed=int(args.dataset_seed),
            disamb_mode="hardened",
            cf_include_shams=True,
            coh_include_controls=True,
            dry_run=bool(args.dry_run),
        )

    smoke_model_dir = results_dir / "local_model"
    smoke_data_dir = results_dir / "smoke_data"
    smoke_manifest_path = smoke_data_dir / "DATASET_MANIFEST.json"
    if not bool(args.dry_run):
        ensure_smoke_local_model(smoke_model_dir)
        smoke_paths = ensure_smoke_datasets(smoke_data_dir)
        files = {
            "disamb_pairs.jsonl": {
                "path": _portable_manifest_path(smoke_paths["disamb"], base_dir=smoke_data_dir.parent),
                "n_lines": int(_count_nonempty_lines(smoke_paths["disamb"])),
                "sha256": _sha256_file(smoke_paths["disamb"]),
            },
            "counterfactual.jsonl": {
                "path": _portable_manifest_path(smoke_paths["cf"], base_dir=smoke_data_dir.parent),
                "n_lines": int(_count_nonempty_lines(smoke_paths["cf"])),
                "sha256": _sha256_file(smoke_paths["cf"]),
            },
            "coherence.jsonl": {
                "path": _portable_manifest_path(smoke_paths["coh"], base_dir=smoke_data_dir.parent),
                "n_lines": int(_count_nonempty_lines(smoke_paths["coh"])),
                "sha256": _sha256_file(smoke_paths["coh"]),
            },
        }
        from aom.data.bundle_manifest import compute_bundle_id

        smoke_manifest = {
            "name": "smoke_bundle",
            "generated_at_utc": _utc_now_iso(),
            "git_commit": _try_git_commit(),
            "generator": "scripts/run_paper.py:ensure_smoke_datasets",
            "files": files,
            "bundle_id": compute_bundle_id({"files": files}),
        }
        _write_json(smoke_manifest_path, smoke_manifest)
    else:
        smoke_paths = {
            "disamb": smoke_data_dir / "disamb_pairs.jsonl",
            "cf": smoke_data_dir / "counterfactual.jsonl",
            "coh": smoke_data_dir / "coherence.jsonl",
        }

    commands: List[List[str]] = []
    csv_path = results_dir / "aom_eval.csv"
    cmd = _aom_eval_cmd(
        models=[str(smoke_model_dir)],
        device="cpu",
        torch_dtype="float32",
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
        disamb_path=str(smoke_paths["disamb"]),
        cf_path=str(smoke_paths["cf"]),
        coh_path=str(smoke_paths["coh"]),
        dataset_manifest_path=str(smoke_manifest_path) if not bool(args.dry_run) else None,
        bootstrap_n=50,
        bootstrap_seed=42,
        ci=0.95,
        run_patching=True,
        patch_layers="0",
        run_patching_specificity=True,
        patch_specificity_depth_frac=0.25,
        patch_specificity_buffer=2,
        patch_specificity_position_window=8,
        patch_specificity_seed=0,
        run_clt_patching=bool(getattr(args, "run_clt_patching", False)),
        clt_repo=str(getattr(args, "clt_repo", "")),
        clt_width=str(getattr(args, "clt_width", "16k")),
        clt_run_name=getattr(args, "clt_run_name", None),
        clt_l0_target=getattr(args, "clt_l0_target", None),
        clt_layers=str(getattr(args, "clt_layers", "")),
        clt_scale=float(getattr(args, "clt_scale", 1.0)),
        clt_dtype=str(getattr(args, "clt_dtype", "float32")),
        clt_decode_strategy=str(getattr(args, "clt_decode_strategy", "delta_1decode")),
        clt_dtype_policy=str(getattr(args, "clt_dtype_policy", "clt")),
        clt_eps_active=float(getattr(args, "clt_eps_active", 1e-6)),
        csv_path=str(csv_path),
    )
    commands.append(cmd)
    _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "run_clt_stage", False)):
        clt_csv = results_dir / "clt_cpt_disamb_only.csv"
        clt_cmd = _aom_eval_cmd(
            models=[str(smoke_model_dir)],
            device="cpu",
            torch_dtype="float32",
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
            disamb_path=str(smoke_paths["disamb"]),
            cf_path="",
            coh_path="",
            dataset_manifest_path=str(smoke_manifest_path) if not bool(args.dry_run) else None,
            bootstrap_n=50,
            bootstrap_seed=42,
            ci=0.95,
            run_clt_patching=True,
            clt_repo=str(getattr(args, "clt_repo", "")),
            clt_width=str(getattr(args, "clt_width", "16k")),
            clt_run_name=getattr(args, "clt_run_name", None),
            clt_l0_target=getattr(args, "clt_l0_target", None),
            clt_layers=str(getattr(args, "clt_layers", "")),
            clt_scale=float(getattr(args, "clt_scale", 1.0)),
            clt_dtype=str(getattr(args, "clt_dtype", "float32")),
            clt_decode_strategy=str(getattr(args, "clt_decode_strategy", "delta_1decode")),
            clt_dtype_policy=str(getattr(args, "clt_dtype_policy", "clt")),
            clt_eps_active=float(getattr(args, "clt_eps_active", 1e-6)),
            csv_path=str(clt_csv),
        )
        commands.append(clt_cmd)
        _run(clt_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "clt_topk_recovery", False)):
        clt_repo = str(getattr(args, "clt_repo", "")).strip()
        if not clt_repo:
            raise ValueError("--clt_topk_recovery requires --clt_repo")
        topk_cmd = _clt_topk_recovery_cmd(
            model_name_or_path=str(smoke_model_dir),
            clt_repo=clt_repo,
            clt_width=str(getattr(args, "clt_width", "16k")),
            clt_run_name=getattr(args, "clt_run_name", None),
            clt_l0_target=getattr(args, "clt_l0_target", None),
            layers=str(getattr(args, "clt_topk_layers", "4,8,12")),
            ks=str(getattr(args, "clt_topk_ks", "1,5,10,20,50,100,200,500,1000,2000,4000,8000,16384")),
            logz_ks=str(getattr(args, "clt_topk_logz_ks", "20,50,200,16384")),
            split_seed=int(getattr(args, "clt_topk_split_seed", 0)),
            frac_selection=float(getattr(args, "clt_topk_frac_selection", 0.5)),
            random_k_seeds=str(getattr(args, "clt_topk_random_k_seeds", "0,1,2,3,4")),
            random_control_mode=str(getattr(args, "clt_topk_random_control_mode", "complement")),
            matched_bin_n_bins=int(getattr(args, "clt_topk_matched_bin_n_bins", 10)),
            eps=float(getattr(args, "clt_topk_eps", 1e-6)),
            bootstrap_B=int(getattr(args, "bootstrap_n", 1000)),
            ci=float(getattr(args, "ci", 0.95)),
            with_logz=bool(getattr(args, "clt_topk_with_logz", False)),
            disamb_path=str(smoke_paths["disamb"]),
            device="cpu",
            torch_dtype="float32",
            local_files_only=True,
            trust_remote_code=False,
            seed=int(getattr(args, "seed", 0)),
            out_csv=str(results_dir / "clt_topk_recovery.csv"),
            out_summary=str(results_dir / "clt_topk_recovery.summary.json"),
        )
        commands.append(topk_cmd)
        _run(topk_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "clt_feature_analysis", False)):
        clt_repo = str(getattr(args, "clt_repo", "")).strip()
        if not clt_repo:
            raise ValueError("--clt_feature_analysis requires --clt_repo")
        topk_summary_path = results_dir / "clt_topk_recovery.summary.json"
        if not bool(args.dry_run) and not topk_summary_path.exists():
            raise FileNotFoundError(
                f"Missing top-k summary for CLT feature analysis: {str(topk_summary_path)}. "
                "Run with --clt_topk_recovery first."
            )
        cmd = _clt_feature_analysis_cmd(
            model_name_or_path=str(smoke_model_dir),
            clt_repo=clt_repo,
            clt_layer=int(getattr(args, "clt_feature_layer", 4)),
            clt_width=str(getattr(args, "clt_width", "16k")),
            clt_run_name=getattr(args, "clt_run_name", None),
            clt_l0_target=getattr(args, "clt_l0_target", None),
            topk_summary_path=str(topk_summary_path),
            top_n_features=int(getattr(args, "clt_feature_top_n_features", 50)),
            max_pairs=int(getattr(args, "clt_feature_max_pairs", 0)),
            disamb_path=str(smoke_paths["disamb"]),
            device="cpu",
            torch_dtype="float32",
            local_files_only=True,
            trust_remote_code=False,
            similarity_threshold=float(getattr(args, "clt_feature_similarity_threshold", 0.8)),
            stability_bootstrap_n=int(getattr(args, "clt_feature_stability_bootstrap_n", 200)),
            null_n=int(getattr(args, "clt_feature_null_n", 500)),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            out_prefix=str(results_dir / "clt_feature_analysis"),
            smoke=True,
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "clt_head_attribution", False)):
        clt_repo = str(getattr(args, "clt_repo", "")).strip()
        if not clt_repo:
            raise ValueError("--clt_head_attribution requires --clt_repo")
        topk_summary_path = results_dir / "clt_topk_recovery.summary.json"
        if not bool(args.dry_run) and not topk_summary_path.exists():
            raise FileNotFoundError(
                f"Missing top-k summary for CLT head attribution: {str(topk_summary_path)}. "
                "Run with --clt_topk_recovery first."
            )
        cmd = _clt_head_attribution_cmd(
            model_name_or_path=str(smoke_model_dir),
            clt_repo=clt_repo,
            clt_layer=int(getattr(args, "clt_feature_layer", 4)),
            clt_width=str(getattr(args, "clt_width", "16k")),
            clt_run_name=getattr(args, "clt_run_name", None),
            clt_l0_target=getattr(args, "clt_l0_target", None),
            topk_summary_path=str(topk_summary_path),
            top_n_features=int(getattr(args, "clt_feature_top_n_features", 50)),
            head_layers=str(getattr(args, "clt_head_layers", "1,2,3")),
            top_h=int(getattr(args, "clt_head_top_h", 5)),
            split_seed=int(getattr(args, "clt_topk_split_seed", 0)),
            frac_selection=float(getattr(args, "clt_topk_frac_selection", 0.5)),
            disamb_path=str(smoke_paths["disamb"]),
            device="cpu",
            torch_dtype="float32",
            local_files_only=True,
            trust_remote_code=False,
            seed=int(getattr(args, "seed", 0)),
            max_pairs=int(getattr(args, "clt_head_max_pairs", 0)),
            ablate_positions=str(getattr(args, "clt_head_ablate_positions", "target")),
            qk_patterns=bool(getattr(args, "clt_head_qk_patterns", False)),
            bootstrap_n=int(getattr(args, "bootstrap_n", 1000)),
            bootstrap_seed=int(getattr(args, "bootstrap_seed", 42)),
            ci=float(getattr(args, "ci", 0.95)),
            out_prefix=str(results_dir / "clt_head_attribution"),
            smoke=True,
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "mechanistic_disamb_path_decomp", False)):
        decomp_cmd = _disamb_path_decomp_cmd(
            model_name_or_path=str(smoke_model_dir),
            device="cpu",
            torch_dtype="float32",
            local_files_only=True,
            trust_remote_code=False,
            disamb_path=str(smoke_paths["disamb"]),
            bootstrap_n=min(int(args.bootstrap_n), 100),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            max_pairs=int(getattr(args, "mechanistic_disamb_max_pairs", 8)),
            mode=str(getattr(args, "mechanistic_disamb_mode", "layer")),
            position=int(getattr(args, "mechanistic_disamb_position", -1)),
            rows_csv_path=str(results_dir / "disamb_path_decomp_rows.csv"),
            summary_csv_path=str(results_dir / "disamb_path_decomp_summary.csv"),
        )
        commands.append(decomp_cmd)
        _run(decomp_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "mechanistic_why_fetch", False)):
        tasks_raw = str(getattr(args, "mechanistic_why_fetch_tasks", "disamb")).strip()
        tasks = [t.strip() for t in tasks_raw.split(",") if t.strip()]
        for task in tasks:
            cmd = _why_fetch_cmd(
                task=str(task),
                model_name_or_path=str(smoke_model_dir),
                device="cpu",
                torch_dtype="float32",
                local_files_only=True,
                trust_remote_code=False,
                disamb_path=str(smoke_paths["disamb"]),
                cf_path=str(smoke_paths["cf"]),
                coh_path=str(smoke_paths["coh"]),
                n_examples=int(getattr(args, "mechanistic_why_fetch_n_examples", 10)),
                heads_topk=int(getattr(args, "mechanistic_why_fetch_heads_topk", 2)),
                bootstrap_n=min(int(args.bootstrap_n), 100),
                bootstrap_seed=int(args.bootstrap_seed),
                ci=float(args.ci),
                rows_csv_path=str(results_dir / f"why_fetch_{task}_rows.csv"),
                summary_csv_path=str(results_dir / f"why_fetch_{task}_summary.csv"),
                smoke=True,
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "sae_feature_families", False)):
        sae_repo = str(getattr(args, "sae_repo", "")).strip()
        if not sae_repo:
            raise ValueError("--sae_feature_families requires --sae_repo")
        ff_cmd = _feature_families_cmd(
            model_name_or_path=str(smoke_model_dir),
            sae_repo=sae_repo,
            sae_layer=int(getattr(args, "sae_layer", 0)),
            sae_width=str(getattr(args, "sae_width", "16k")),
            sae_scale=float(getattr(args, "sae_scale", 1.0)),
            n_features=int(getattr(args, "sae_n_features", 16)),
            max_pairs=int(getattr(args, "sae_max_pairs", 8)),
            similarity_threshold=float(getattr(args, "sae_similarity_threshold", 0.8)),
            null_n=int(getattr(args, "sae_null_n", 100)),
            disamb_path=str(smoke_paths["disamb"]),
            device="cpu",
            torch_dtype="float32",
            local_files_only=True,
            trust_remote_code=False,
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            out_prefix=str(results_dir / "smoke"),
            smoke=True,
        )
        commands.append(ff_cmd)
        _run(ff_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "completeness", False)):
        comp_cmd = _completeness_cmd(
            model_name_or_path=str(smoke_model_dir),
            tasks=str(getattr(args, "completeness_tasks", "disamb,cf,coh")),
            disamb_path=str(smoke_paths["disamb"]),
            cf_path=str(smoke_paths["cf"]),
            coh_path=str(smoke_paths["coh"]),
            layers=str(getattr(args, "completeness_layers", "")),
            explanation_path=str(getattr(args, "completeness_explanation_path", "")),
            device="cpu",
            torch_dtype="float32",
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
            bootstrap_n=min(int(args.bootstrap_n), 100),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            csv_path=str(results_dir / "completeness.csv"),
            smoke=True,
        )
        commands.append(comp_cmd)
        _run(comp_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    report_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "report_results.py"),
        "--results_dir",
        str(results_dir),
        "--out_path",
        str(results_dir / "results_report.md"),
    ]
    commands.append(report_cmd)
    _run(report_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    run = PaperRun(mode="smoke", results_dir=results_dir, dataset_manifest_path=dataset_manifest_path, commands=commands)
    if not bool(args.dry_run):
        _write_run_manifest(run)
    return run


def _run_m1max_behavioral_split(
    *,
    results_dir: Path,
    dataset_manifest_path: Optional[Path],
    models: List[str],
    disamb_path: Path,
    cf_path: Path,
    coh_path: Path,
    commands: List[List[str]],
    args: argparse.Namespace,
    attn_implementation: str,
    torch_dtype: str,
    timeout_seconds: int,
) -> List[BehavioralModelResult]:
    behavioral_dir = results_dir / "behavioral"
    behavioral_dir.mkdir(parents=True, exist_ok=True)
    results: List[BehavioralModelResult] = []
    successful_csvs: List[Path] = []
    for model_name in models:
        model_csv = behavioral_dir / f"{_model_slug(model_name)}.csv"
        cmd = _aom_eval_cmd(
            models=[str(model_name)],
            device="mps",
            torch_dtype=str(torch_dtype),
            attn_implementation=str(attn_implementation),
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            disamb_path=str(disamb_path),
            cf_path=str(cf_path),
            coh_path=str(coh_path),
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            run_clt_patching=bool(getattr(args, "run_clt_patching", False)),
            clt_repo=str(getattr(args, "clt_repo", "")),
            clt_width=str(getattr(args, "clt_width", "16k")),
            clt_run_name=getattr(args, "clt_run_name", None),
            clt_l0_target=getattr(args, "clt_l0_target", None),
            clt_layers=str(getattr(args, "clt_layers", "")),
            clt_scale=float(getattr(args, "clt_scale", 1.0)),
            clt_dtype=str(getattr(args, "clt_dtype", "float32")),
            clt_decode_strategy=str(getattr(args, "clt_decode_strategy", "delta_1decode")),
            clt_dtype_policy=str(getattr(args, "clt_dtype_policy", "clt")),
            clt_eps_active=float(getattr(args, "clt_eps_active", 1e-6)),
            csv_path=str(model_csv),
            device_map=str(args.device_map) if args.device_map else None,
        )
        commands.append(cmd)
        outcome = _run_with_timeout(
            cmd,
            cwd=ROOT,
            dry_run=bool(args.dry_run),
            timeout_seconds=int(timeout_seconds),
        )
        if bool(args.dry_run):
            continue
        if outcome.timed_out:
            results.append(
                BehavioralModelResult(
                    model=str(model_name),
                    status="TIMEOUT",
                    csv_path=str(model_csv),
                    reason=f"timed out after {int(timeout_seconds)}s",
                    exit_code=int(outcome.returncode),
                )
            )
            continue
        if int(outcome.returncode) != 0:
            results.append(
                BehavioralModelResult(
                    model=str(model_name),
                    status="FAIL",
                    csv_path=str(model_csv),
                    reason=f"command exit {int(outcome.returncode)}",
                    exit_code=int(outcome.returncode),
                )
            )
            continue
        if not model_csv.exists() or model_csv.stat().st_size == 0:
            results.append(
                BehavioralModelResult(
                    model=str(model_name),
                    status="FAIL",
                    csv_path=str(model_csv),
                    reason="missing behavioral csv output",
                )
            )
            continue
        successful_csvs.append(model_csv)
        results.append(
            BehavioralModelResult(
                model=str(model_name),
                status="PASS",
                csv_path=str(model_csv),
            )
        )
    if not bool(args.dry_run):
        _merge_csv_files(successful_csvs, results_dir / "aom_eval.csv")
    return results


def _run_m1max_impl(
    args: argparse.Namespace,
    *,
    mode_name: str,
    behavioral_attn: str,
    split_behavioral: bool,
) -> PaperRun:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.data_dir)
    dataset_manifest_path: Optional[Path] = None
    if not bool(args.skip_dataset):
        dataset_manifest_path = ensure_paper_dataset(
            data_dir=dataset_dir,
            seed=int(args.dataset_seed),
            disamb_mode="hardened",
            cf_include_shams=True,
            coh_include_controls=True,
            dry_run=bool(args.dry_run),
        )
    else:
        p = dataset_dir / "DATASET_MANIFEST.json"
        dataset_manifest_path = p if p.exists() else None

    models = list(M1MAX_MODELS)
    commands: List[List[str]] = []

    disamb_path = dataset_dir / "disamb_pairs.jsonl"
    cf_path = dataset_dir / "counterfactual.jsonl"
    coh_path = dataset_dir / "coherence.jsonl"
    mps_safe_eval_torch_dtype = "float32" if bool(split_behavioral) else "float16"

    behavioral_results: List[BehavioralModelResult] = []
    if split_behavioral:
        behavioral_results = _run_m1max_behavioral_split(
            results_dir=results_dir,
            dataset_manifest_path=dataset_manifest_path,
            models=models,
            disamb_path=disamb_path,
            cf_path=cf_path,
            coh_path=coh_path,
            commands=commands,
            args=args,
            attn_implementation=str(behavioral_attn),
            torch_dtype=str(mps_safe_eval_torch_dtype),
            timeout_seconds=int(getattr(args, "model_timeout_seconds", 1200)),
        )
    else:
        behavioral_csv = results_dir / "aom_eval.csv"
        cmd = _aom_eval_cmd(
            models=models,
            device="mps",
            torch_dtype="float16",
            attn_implementation=str(behavioral_attn),
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            disamb_path=str(disamb_path),
            cf_path=str(cf_path),
            coh_path=str(coh_path),
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            run_clt_patching=bool(getattr(args, "run_clt_patching", False)),
            clt_repo=str(getattr(args, "clt_repo", "")),
            clt_width=str(getattr(args, "clt_width", "16k")),
            clt_run_name=getattr(args, "clt_run_name", None),
            clt_l0_target=getattr(args, "clt_l0_target", None),
            clt_layers=str(getattr(args, "clt_layers", "")),
            clt_scale=float(getattr(args, "clt_scale", 1.0)),
            clt_dtype=str(getattr(args, "clt_dtype", "float32")),
            clt_decode_strategy=str(getattr(args, "clt_decode_strategy", "delta_1decode")),
            clt_dtype_policy=str(getattr(args, "clt_dtype_policy", "clt")),
            clt_eps_active=float(getattr(args, "clt_eps_active", 1e-6)),
            csv_path=str(behavioral_csv),
            device_map=str(args.device_map) if args.device_map else None,
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "run_clt_stage", False)):
        clt_csv = results_dir / "clt_cpt_disamb_only.csv"
        clt_cmd = _aom_eval_cmd(
            models=models,
            device="mps",
            torch_dtype="float16",
            attn_implementation="eager",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            disamb_path=str(disamb_path),
            cf_path="",
            coh_path="",
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            run_clt_patching=True,
            clt_repo=str(getattr(args, "clt_repo", "")),
            clt_width=str(getattr(args, "clt_width", "16k")),
            clt_run_name=getattr(args, "clt_run_name", None),
            clt_l0_target=getattr(args, "clt_l0_target", None),
            clt_layers=str(getattr(args, "clt_layers", "")),
            clt_scale=float(getattr(args, "clt_scale", 1.0)),
            clt_dtype=str(getattr(args, "clt_dtype", "float32")),
            clt_decode_strategy=str(getattr(args, "clt_decode_strategy", "delta_1decode")),
            clt_dtype_policy=str(getattr(args, "clt_dtype_policy", "clt")),
            clt_eps_active=float(getattr(args, "clt_eps_active", 1e-6)),
            csv_path=str(clt_csv),
        )
        commands.append(clt_cmd)
        _run(clt_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "clt_topk_recovery", False)):
        clt_repo = str(getattr(args, "clt_repo", "")).strip()
        if not clt_repo:
            raise ValueError("--clt_topk_recovery requires --clt_repo")
        for m in [str(x) for x in models]:
            safe = str(m).replace("/", "_")
            cmd = _clt_topk_recovery_cmd(
                model_name_or_path=str(m),
                clt_repo=clt_repo,
                clt_width=str(getattr(args, "clt_width", "16k")),
                clt_run_name=getattr(args, "clt_run_name", None),
                clt_l0_target=getattr(args, "clt_l0_target", None),
                layers=str(getattr(args, "clt_topk_layers", "4,8,12")),
                ks=str(getattr(args, "clt_topk_ks", "1,5,10,20,50,100,200,500,1000,2000,4000,8000,16384")),
                logz_ks=str(getattr(args, "clt_topk_logz_ks", "20,50,200,16384")),
                split_seed=int(getattr(args, "clt_topk_split_seed", 0)),
                frac_selection=float(getattr(args, "clt_topk_frac_selection", 0.5)),
                random_k_seeds=str(getattr(args, "clt_topk_random_k_seeds", "0,1,2,3,4")),
                random_control_mode=str(getattr(args, "clt_topk_random_control_mode", "complement")),
                matched_bin_n_bins=int(getattr(args, "clt_topk_matched_bin_n_bins", 10)),
                eps=float(getattr(args, "clt_topk_eps", 1e-6)),
                bootstrap_B=int(getattr(args, "bootstrap_n", 1000)),
                ci=float(getattr(args, "ci", 0.95)),
                with_logz=bool(getattr(args, "clt_topk_with_logz", False)),
                disamb_path=str(disamb_path),
                device="mps",
                torch_dtype="float16",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                seed=int(getattr(args, "seed", 0)),
                out_csv=str(results_dir / f"clt_topk_recovery_{safe}.csv"),
                out_summary=str(results_dir / f"clt_topk_recovery_{safe}.summary.json"),
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "clt_feature_analysis", False)):
        clt_repo = str(getattr(args, "clt_repo", "")).strip()
        if not clt_repo:
            raise ValueError("--clt_feature_analysis requires --clt_repo")
        for m in [str(x) for x in models]:
            safe = str(m).replace("/", "_")
            topk_summary_path = results_dir / f"clt_topk_recovery_{safe}.summary.json"
            if not bool(args.dry_run) and not topk_summary_path.exists():
                raise FileNotFoundError(
                    f"Missing top-k summary for CLT feature analysis: {str(topk_summary_path)}. "
                    "Run with --clt_topk_recovery first."
                )
            cmd = _clt_feature_analysis_cmd(
                model_name_or_path=str(m),
                clt_repo=clt_repo,
                clt_layer=int(getattr(args, "clt_feature_layer", 4)),
                clt_width=str(getattr(args, "clt_width", "16k")),
                clt_run_name=getattr(args, "clt_run_name", None),
                clt_l0_target=getattr(args, "clt_l0_target", None),
                topk_summary_path=str(topk_summary_path),
                top_n_features=int(getattr(args, "clt_feature_top_n_features", 50)),
                max_pairs=int(getattr(args, "clt_feature_max_pairs", 0)),
                disamb_path=str(disamb_path),
                device="mps",
                torch_dtype="float16",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                similarity_threshold=float(getattr(args, "clt_feature_similarity_threshold", 0.8)),
                stability_bootstrap_n=int(getattr(args, "clt_feature_stability_bootstrap_n", 200)),
                null_n=int(getattr(args, "clt_feature_null_n", 500)),
                bootstrap_seed=int(args.bootstrap_seed),
                ci=float(args.ci),
                out_prefix=str(results_dir / f"clt_feature_analysis_{safe}"),
                smoke=False,
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "clt_head_attribution", False)):
        clt_repo = str(getattr(args, "clt_repo", "")).strip()
        if not clt_repo:
            raise ValueError("--clt_head_attribution requires --clt_repo")
        for m in [str(x) for x in models]:
            safe = str(m).replace("/", "_")
            topk_summary_path = results_dir / f"clt_topk_recovery_{safe}.summary.json"
            if not bool(args.dry_run) and not topk_summary_path.exists():
                raise FileNotFoundError(
                    f"Missing top-k summary for CLT head attribution: {str(topk_summary_path)}. "
                    "Run with --clt_topk_recovery first."
                )
            cmd = _clt_head_attribution_cmd(
                model_name_or_path=str(m),
                clt_repo=clt_repo,
                clt_layer=int(getattr(args, "clt_feature_layer", 4)),
                clt_width=str(getattr(args, "clt_width", "16k")),
                clt_run_name=getattr(args, "clt_run_name", None),
                clt_l0_target=getattr(args, "clt_l0_target", None),
                topk_summary_path=str(topk_summary_path),
                top_n_features=int(getattr(args, "clt_feature_top_n_features", 50)),
                head_layers=str(getattr(args, "clt_head_layers", "1,2,3")),
                top_h=int(getattr(args, "clt_head_top_h", 5)),
                split_seed=int(getattr(args, "clt_topk_split_seed", 0)),
                frac_selection=float(getattr(args, "clt_topk_frac_selection", 0.5)),
                disamb_path=str(disamb_path),
                device="mps",
                torch_dtype="float16",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                seed=int(getattr(args, "seed", 0)),
                max_pairs=int(getattr(args, "clt_head_max_pairs", 0)),
                ablate_positions=str(getattr(args, "clt_head_ablate_positions", "target")),
                qk_patterns=bool(getattr(args, "clt_head_qk_patterns", False)),
                bootstrap_n=int(getattr(args, "bootstrap_n", 1000)),
                bootstrap_seed=int(getattr(args, "bootstrap_seed", 42)),
                ci=float(getattr(args, "ci", 0.95)),
                out_prefix=str(results_dir / f"clt_head_attribution_{safe}"),
                smoke=False,
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    # 1b) CF/COH causal patching (eager attention required).
    if not bool(getattr(args, "skip_cf_patching", False)):
        cf_patching_csv = results_dir / "cf_patching.csv"
        cmd = _cf_patching_cmd(
            models=list(CF_PATCHING_MODELS),
            device="mps",
            torch_dtype="float16",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            cf_path=str(cf_path),
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            csv_path=str(cf_patching_csv),
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))
    if not bool(getattr(args, "skip_coh_patching", False)):
        coh_patching_csv = results_dir / "coh_patching.csv"
        cmd = _coh_patching_cmd(
            models=list(COH_PATCHING_MODELS),
            device="mps",
            torch_dtype="float16",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            coh_path=str(coh_path),
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            csv_path=str(coh_patching_csv),
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    # 2) CPT layer sweep on "small" models only (DISAMB-only, eager attention required).
    if not bool(args.skip_patching):
        sweep_models = ["gpt2", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]
        sweep_csv = results_dir / "cpt_layer_sweep_disamb_only.csv"
        cmd = _aom_eval_cmd(
            models=sweep_models,
            device="mps",
            torch_dtype=str(mps_safe_eval_torch_dtype),
            attn_implementation="eager",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            disamb_path=str(disamb_path),
            cf_path="",
            coh_path="",
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            csv_path=str(sweep_csv),
            run_patching=True,
            patch_layers="",
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    # 3) CPT target-specificity control (DISAMB-only, fixed depth).
    if not bool(args.skip_specificity):
        seeds = _parse_int_csv(str(getattr(args, "specificity_selection_seeds", "0")))
        if not seeds:
            seeds = [0]
        for spec_seed in seeds:
            if len(seeds) > 1:
                spec_csv = results_dir / f"cpt_specificity_seed{int(spec_seed)}_disamb_only.csv"
            else:
                spec_csv = results_dir / "cpt_specificity_disamb_only.csv"
            cmd = _aom_eval_cmd(
                models=models,
                device="mps",
                torch_dtype=str(mps_safe_eval_torch_dtype),
                attn_implementation="eager",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                revision=getattr(args, "revision", None),
                tokenizer_revision=getattr(args, "tokenizer_revision", None),
                disamb_path=str(disamb_path),
                cf_path="",
                coh_path="",
                dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
                bootstrap_n=int(args.bootstrap_n),
                bootstrap_seed=int(args.bootstrap_seed),
                ci=float(args.ci),
                csv_path=str(spec_csv),
                run_patching_specificity=True,
                patch_specificity_depth_frac=float(getattr(args, "specificity_depth_frac", 0.25)),
                patch_specificity_buffer=int(getattr(args, "specificity_buffer", 2)),
                patch_specificity_position_window=int(getattr(args, "specificity_position_window", 8)),
                patch_specificity_seed=int(spec_seed),
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "mechanistic_disamb_path_decomp", False)):
        mech_models = [str(m) for m in models]
        for m in mech_models:
            safe = str(m).replace("/", "_")
            cmd = _disamb_path_decomp_cmd(
                model_name_or_path=str(m),
                device="mps",
                torch_dtype="float16",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                disamb_path=str(disamb_path),
                bootstrap_n=int(args.bootstrap_n),
                bootstrap_seed=int(args.bootstrap_seed),
                ci=float(args.ci),
                max_pairs=int(getattr(args, "mechanistic_disamb_max_pairs", 0)),
                mode=str(getattr(args, "mechanistic_disamb_mode", "layer")),
                position=int(getattr(args, "mechanistic_disamb_position", -1)),
                rows_csv_path=str(results_dir / f"disamb_path_decomp_rows_{safe}.csv"),
                summary_csv_path=str(results_dir / f"disamb_path_decomp_summary_{safe}.csv"),
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "mechanistic_why_fetch", False)):
        tasks_raw = str(getattr(args, "mechanistic_why_fetch_tasks", "disamb")).strip()
        tasks = [t.strip() for t in tasks_raw.split(",") if t.strip()]
        for m in [str(x) for x in models]:
            safe = str(m).replace("/", "_")
            for task in tasks:
                cmd = _why_fetch_cmd(
                    task=str(task),
                    model_name_or_path=str(m),
                    device="mps",
                    torch_dtype="float16",
                    local_files_only=bool(args.local_files_only),
                    trust_remote_code=bool(args.trust_remote_code),
                    disamb_path=str(disamb_path),
                    cf_path=str(cf_path),
                    coh_path=str(coh_path),
                    n_examples=int(getattr(args, "mechanistic_why_fetch_n_examples", 100)),
                    heads_topk=int(getattr(args, "mechanistic_why_fetch_heads_topk", 4)),
                    bootstrap_n=int(args.bootstrap_n),
                    bootstrap_seed=int(args.bootstrap_seed),
                    ci=float(args.ci),
                    rows_csv_path=str(results_dir / f"why_fetch_{task}_rows_{safe}.csv"),
                    summary_csv_path=str(results_dir / f"why_fetch_{task}_summary_{safe}.csv"),
                    smoke=False,
                )
                commands.append(cmd)
                _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "sae_feature_families", False)):
        sae_repo = str(getattr(args, "sae_repo", "")).strip()
        if not sae_repo:
            raise ValueError("--sae_feature_families requires --sae_repo")
        for m in [str(x) for x in models]:
            safe = str(m).replace("/", "_")
            cmd = _feature_families_cmd(
                model_name_or_path=str(m),
                sae_repo=sae_repo,
                sae_layer=int(getattr(args, "sae_layer", 15)),
                sae_width=str(getattr(args, "sae_width", "16k")),
                sae_scale=float(getattr(args, "sae_scale", 1.0)),
                n_features=int(getattr(args, "sae_n_features", 64)),
                max_pairs=int(getattr(args, "sae_max_pairs", 32)),
                similarity_threshold=float(getattr(args, "sae_similarity_threshold", 0.8)),
                null_n=int(getattr(args, "sae_null_n", 500)),
                disamb_path=str(disamb_path),
                device="mps",
                torch_dtype="float16",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                bootstrap_seed=int(args.bootstrap_seed),
                ci=float(args.ci),
                out_prefix=str(results_dir / f"feature_families_{safe}"),
                smoke=False,
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "completeness", False)):
        comp_tasks = str(getattr(args, "completeness_tasks", "disamb,cf,coh"))
        comp_layers = str(getattr(args, "completeness_layers", ""))
        comp_expl = str(getattr(args, "completeness_explanation_path", ""))
        for m in [str(x) for x in models]:
            safe = str(m).replace("/", "_")
            comp_cmd = _completeness_cmd(
                model_name_or_path=str(m),
                tasks=comp_tasks,
                disamb_path=str(disamb_path),
                cf_path=str(cf_path),
                coh_path=str(coh_path),
                layers=comp_layers,
                explanation_path=comp_expl,
                device="mps",
                torch_dtype="float16",
                attn_implementation="eager",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                bootstrap_n=int(args.bootstrap_n),
                bootstrap_seed=int(args.bootstrap_seed),
                ci=float(args.ci),
                csv_path=str(results_dir / f"completeness_{safe}.csv"),
                smoke=False,
            )
            commands.append(comp_cmd)
            _run(comp_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    report_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "report_results.py"),
        "--results_dir",
        str(results_dir),
        "--out_path",
        str(results_dir / "results_report.md"),
    ]
    commands.append(report_cmd)
    _run(report_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    behavioral_failures = [r for r in behavioral_results if r.status != "PASS"]
    if split_behavioral and not bool(args.dry_run):
        md_path, _json_path = _write_behavioral_status_artifacts(results_dir, behavioral_results)
        _append_markdown_to_report(results_dir / "results_report.md", md_path.read_text(encoding="utf-8"))

    run = PaperRun(mode=str(mode_name), results_dir=results_dir, dataset_manifest_path=dataset_manifest_path, commands=commands)
    if not bool(args.dry_run):
        _write_run_manifest(run)
    if behavioral_failures:
        failed_models = ", ".join(str(r.model) for r in behavioral_failures)
        raise RuntimeError(f"MPS behavioral stage had model failures: {failed_models}")
    return run


def run_m1max(args: argparse.Namespace) -> PaperRun:
    return _run_m1max_impl(
        args,
        mode_name="m1max",
        behavioral_attn="sdpa",
        split_behavioral=False,
    )


def run_m1max_safe(args: argparse.Namespace) -> PaperRun:
    return _run_m1max_impl(
        args,
        mode_name="m1max_safe",
        behavioral_attn="eager",
        split_behavioral=True,
    )


def run_a100(args: argparse.Namespace) -> PaperRun:
    if bool(args.device_map) and not bool(args.skip_patching):
        raise ValueError("For patching runs, avoid --device_map (patching assumes a single-device model). Use --skip_patching.")
    if bool(args.device_map) and not bool(args.skip_specificity):
        raise ValueError(
            "For patching-specificity runs, avoid --device_map (patching assumes a single-device model). Use --skip_specificity."
        )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.data_dir)
    dataset_manifest_path: Optional[Path] = None
    if not bool(args.skip_dataset):
        dataset_manifest_path = ensure_paper_dataset(
            data_dir=dataset_dir,
            seed=int(args.dataset_seed),
            disamb_mode="hardened",
            cf_include_shams=True,
            coh_include_controls=True,
            dry_run=bool(args.dry_run),
        )
    else:
        p = dataset_dir / "DATASET_MANIFEST.json"
        dataset_manifest_path = p if p.exists() else None

    models = list(M1MAX_MODELS) + list(A100_EXTRA_MODELS)
    commands: List[List[str]] = []

    disamb_path = dataset_dir / "disamb_pairs.jsonl"
    cf_path = dataset_dir / "counterfactual.jsonl"
    coh_path = dataset_dir / "coherence.jsonl"

    # 1) Full AoM behavioral suite (fast attention encouraged).
    behavioral_csv = results_dir / "aom_eval.csv"
    cmd = _aom_eval_cmd(
        models=models,
        device="cuda",
        torch_dtype="bfloat16",
        attn_implementation=str(args.attn_behavioral),
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        revision=getattr(args, "revision", None),
        tokenizer_revision=getattr(args, "tokenizer_revision", None),
        disamb_path=str(disamb_path),
        cf_path=str(cf_path),
        coh_path=str(coh_path),
        dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
        bootstrap_n=int(args.bootstrap_n),
        bootstrap_seed=int(args.bootstrap_seed),
        ci=float(args.ci),
        run_clt_patching=bool(getattr(args, "run_clt_patching", False)),
        clt_repo=str(getattr(args, "clt_repo", "")),
        clt_width=str(getattr(args, "clt_width", "16k")),
        clt_run_name=getattr(args, "clt_run_name", None),
        clt_l0_target=getattr(args, "clt_l0_target", None),
        clt_layers=str(getattr(args, "clt_layers", "")),
        clt_scale=float(getattr(args, "clt_scale", 1.0)),
        clt_dtype=str(getattr(args, "clt_dtype", "float32")),
        clt_decode_strategy=str(getattr(args, "clt_decode_strategy", "delta_1decode")),
        clt_dtype_policy=str(getattr(args, "clt_dtype_policy", "clt")),
        clt_eps_active=float(getattr(args, "clt_eps_active", 1e-6)),
        csv_path=str(behavioral_csv),
        device_map=str(args.device_map) if args.device_map else None,
    )
    commands.append(cmd)
    _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "run_clt_stage", False)):
        clt_csv = results_dir / "clt_cpt_disamb_only.csv"
        clt_cmd = _aom_eval_cmd(
            models=models,
            device="cuda",
            torch_dtype="bfloat16",
            attn_implementation="eager",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            disamb_path=str(disamb_path),
            cf_path="",
            coh_path="",
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            run_clt_patching=True,
            clt_repo=str(getattr(args, "clt_repo", "")),
            clt_width=str(getattr(args, "clt_width", "16k")),
            clt_run_name=getattr(args, "clt_run_name", None),
            clt_l0_target=getattr(args, "clt_l0_target", None),
            clt_layers=str(getattr(args, "clt_layers", "")),
            clt_scale=float(getattr(args, "clt_scale", 1.0)),
            clt_dtype=str(getattr(args, "clt_dtype", "float32")),
            clt_decode_strategy=str(getattr(args, "clt_decode_strategy", "delta_1decode")),
            clt_dtype_policy=str(getattr(args, "clt_dtype_policy", "clt")),
            clt_eps_active=float(getattr(args, "clt_eps_active", 1e-6)),
            csv_path=str(clt_csv),
        )
        commands.append(clt_cmd)
        _run(clt_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "clt_topk_recovery", False)):
        clt_repo = str(getattr(args, "clt_repo", "")).strip()
        if not clt_repo:
            raise ValueError("--clt_topk_recovery requires --clt_repo")
        for m in [str(x) for x in models]:
            safe = str(m).replace("/", "_")
            cmd = _clt_topk_recovery_cmd(
                model_name_or_path=str(m),
                clt_repo=clt_repo,
                clt_width=str(getattr(args, "clt_width", "16k")),
                clt_run_name=getattr(args, "clt_run_name", None),
                clt_l0_target=getattr(args, "clt_l0_target", None),
                layers=str(getattr(args, "clt_topk_layers", "4,8,12")),
                ks=str(getattr(args, "clt_topk_ks", "1,5,10,20,50,100,200,500,1000,2000,4000,8000,16384")),
                logz_ks=str(getattr(args, "clt_topk_logz_ks", "20,50,200,16384")),
                split_seed=int(getattr(args, "clt_topk_split_seed", 0)),
                frac_selection=float(getattr(args, "clt_topk_frac_selection", 0.5)),
                random_k_seeds=str(getattr(args, "clt_topk_random_k_seeds", "0,1,2,3,4")),
                random_control_mode=str(getattr(args, "clt_topk_random_control_mode", "complement")),
                matched_bin_n_bins=int(getattr(args, "clt_topk_matched_bin_n_bins", 10)),
                eps=float(getattr(args, "clt_topk_eps", 1e-6)),
                bootstrap_B=int(getattr(args, "bootstrap_n", 1000)),
                ci=float(getattr(args, "ci", 0.95)),
                with_logz=bool(getattr(args, "clt_topk_with_logz", False)),
                disamb_path=str(disamb_path),
                device="cuda",
                torch_dtype="bfloat16",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                seed=int(getattr(args, "seed", 0)),
                out_csv=str(results_dir / f"clt_topk_recovery_{safe}.csv"),
                out_summary=str(results_dir / f"clt_topk_recovery_{safe}.summary.json"),
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "clt_feature_analysis", False)):
        clt_repo = str(getattr(args, "clt_repo", "")).strip()
        if not clt_repo:
            raise ValueError("--clt_feature_analysis requires --clt_repo")
        for m in [str(x) for x in models]:
            safe = str(m).replace("/", "_")
            topk_summary_path = results_dir / f"clt_topk_recovery_{safe}.summary.json"
            if not bool(args.dry_run) and not topk_summary_path.exists():
                raise FileNotFoundError(
                    f"Missing top-k summary for CLT feature analysis: {str(topk_summary_path)}. "
                    "Run with --clt_topk_recovery first."
                )
            cmd = _clt_feature_analysis_cmd(
                model_name_or_path=str(m),
                clt_repo=clt_repo,
                clt_layer=int(getattr(args, "clt_feature_layer", 4)),
                clt_width=str(getattr(args, "clt_width", "16k")),
                clt_run_name=getattr(args, "clt_run_name", None),
                clt_l0_target=getattr(args, "clt_l0_target", None),
                topk_summary_path=str(topk_summary_path),
                top_n_features=int(getattr(args, "clt_feature_top_n_features", 50)),
                max_pairs=int(getattr(args, "clt_feature_max_pairs", 0)),
                disamb_path=str(disamb_path),
                device="cuda",
                torch_dtype="bfloat16",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                similarity_threshold=float(getattr(args, "clt_feature_similarity_threshold", 0.8)),
                stability_bootstrap_n=int(getattr(args, "clt_feature_stability_bootstrap_n", 200)),
                null_n=int(getattr(args, "clt_feature_null_n", 500)),
                bootstrap_seed=int(args.bootstrap_seed),
                ci=float(args.ci),
                out_prefix=str(results_dir / f"clt_feature_analysis_{safe}"),
                smoke=False,
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "clt_head_attribution", False)):
        clt_repo = str(getattr(args, "clt_repo", "")).strip()
        if not clt_repo:
            raise ValueError("--clt_head_attribution requires --clt_repo")
        for m in [str(x) for x in models]:
            safe = str(m).replace("/", "_")
            topk_summary_path = results_dir / f"clt_topk_recovery_{safe}.summary.json"
            if not bool(args.dry_run) and not topk_summary_path.exists():
                raise FileNotFoundError(
                    f"Missing top-k summary for CLT head attribution: {str(topk_summary_path)}. "
                    "Run with --clt_topk_recovery first."
                )
            cmd = _clt_head_attribution_cmd(
                model_name_or_path=str(m),
                clt_repo=clt_repo,
                clt_layer=int(getattr(args, "clt_feature_layer", 4)),
                clt_width=str(getattr(args, "clt_width", "16k")),
                clt_run_name=getattr(args, "clt_run_name", None),
                clt_l0_target=getattr(args, "clt_l0_target", None),
                topk_summary_path=str(topk_summary_path),
                top_n_features=int(getattr(args, "clt_feature_top_n_features", 50)),
                head_layers=str(getattr(args, "clt_head_layers", "1,2,3")),
                top_h=int(getattr(args, "clt_head_top_h", 5)),
                split_seed=int(getattr(args, "clt_topk_split_seed", 0)),
                frac_selection=float(getattr(args, "clt_topk_frac_selection", 0.5)),
                disamb_path=str(disamb_path),
                device="cuda",
                torch_dtype="bfloat16",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                seed=int(getattr(args, "seed", 0)),
                max_pairs=int(getattr(args, "clt_head_max_pairs", 0)),
                ablate_positions=str(getattr(args, "clt_head_ablate_positions", "target")),
                qk_patterns=bool(getattr(args, "clt_head_qk_patterns", False)),
                bootstrap_n=int(getattr(args, "bootstrap_n", 1000)),
                bootstrap_seed=int(getattr(args, "bootstrap_seed", 42)),
                ci=float(getattr(args, "ci", 0.95)),
                out_prefix=str(results_dir / f"clt_head_attribution_{safe}"),
                smoke=False,
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    # 1b) CF/COH causal patching (eager attention required).
    if not bool(getattr(args, "skip_cf_patching", False)):
        cf_patching_csv = results_dir / "cf_patching.csv"
        cmd = _cf_patching_cmd(
            models=list(CF_PATCHING_MODELS),
            device="cuda",
            torch_dtype="bfloat16",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            cf_path=str(cf_path),
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            csv_path=str(cf_patching_csv),
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))
    if not bool(getattr(args, "skip_coh_patching", False)):
        coh_patching_csv = results_dir / "coh_patching.csv"
        cmd = _coh_patching_cmd(
            models=list(COH_PATCHING_MODELS),
            device="cuda",
            torch_dtype="bfloat16",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            coh_path=str(coh_path),
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            csv_path=str(coh_patching_csv),
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    # 2) CPT layer sweep across all models (DISAMB-only, eager attention required).
    if not bool(args.skip_patching):
        sweep_csv = results_dir / "cpt_layer_sweep_disamb_only.csv"
        cmd = _aom_eval_cmd(
            models=models,
            device="cuda",
            torch_dtype="bfloat16",
            attn_implementation="eager",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            disamb_path=str(disamb_path),
            cf_path="",
            coh_path="",
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            csv_path=str(sweep_csv),
            run_patching=True,
            patch_layers="",
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    # 3) CPT target-specificity control (DISAMB-only).
    if not bool(args.skip_specificity):
        seeds = _parse_int_csv(str(getattr(args, "specificity_selection_seeds", "0")))
        if not seeds:
            seeds = [0]
        for spec_seed in seeds:
            if len(seeds) > 1:
                spec_csv = results_dir / f"cpt_specificity_seed{int(spec_seed)}_disamb_only.csv"
            else:
                spec_csv = results_dir / "cpt_specificity_disamb_only.csv"
            cmd = _aom_eval_cmd(
                models=models,
                device="cuda",
                torch_dtype="bfloat16",
                attn_implementation="eager",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                revision=getattr(args, "revision", None),
                tokenizer_revision=getattr(args, "tokenizer_revision", None),
                disamb_path=str(disamb_path),
                cf_path="",
                coh_path="",
                dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
                bootstrap_n=int(args.bootstrap_n),
                bootstrap_seed=int(args.bootstrap_seed),
                ci=float(args.ci),
                csv_path=str(spec_csv),
                run_patching_specificity=True,
                patch_specificity_depth_frac=float(getattr(args, "specificity_depth_frac", 0.25)),
                patch_specificity_buffer=int(getattr(args, "specificity_buffer", 2)),
                patch_specificity_position_window=int(getattr(args, "specificity_position_window", 8)),
                patch_specificity_seed=int(spec_seed),
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "mechanistic_disamb_path_decomp", False)):
        mech_models = [str(m) for m in models]
        for m in mech_models:
            safe = str(m).replace("/", "_")
            cmd = _disamb_path_decomp_cmd(
                model_name_or_path=str(m),
                device="cuda",
                torch_dtype="bfloat16",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                disamb_path=str(disamb_path),
                bootstrap_n=int(args.bootstrap_n),
                bootstrap_seed=int(args.bootstrap_seed),
                ci=float(args.ci),
                max_pairs=int(getattr(args, "mechanistic_disamb_max_pairs", 0)),
                mode=str(getattr(args, "mechanistic_disamb_mode", "layer")),
                position=int(getattr(args, "mechanistic_disamb_position", -1)),
                rows_csv_path=str(results_dir / f"disamb_path_decomp_rows_{safe}.csv"),
                summary_csv_path=str(results_dir / f"disamb_path_decomp_summary_{safe}.csv"),
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "mechanistic_why_fetch", False)):
        tasks_raw = str(getattr(args, "mechanistic_why_fetch_tasks", "disamb")).strip()
        tasks = [t.strip() for t in tasks_raw.split(",") if t.strip()]
        for m in [str(x) for x in models]:
            safe = str(m).replace("/", "_")
            for task in tasks:
                cmd = _why_fetch_cmd(
                    task=str(task),
                    model_name_or_path=str(m),
                    device="cuda",
                    torch_dtype="bfloat16",
                    local_files_only=bool(args.local_files_only),
                    trust_remote_code=bool(args.trust_remote_code),
                    disamb_path=str(disamb_path),
                    cf_path=str(cf_path),
                    coh_path=str(coh_path),
                    n_examples=int(getattr(args, "mechanistic_why_fetch_n_examples", 100)),
                    heads_topk=int(getattr(args, "mechanistic_why_fetch_heads_topk", 4)),
                    bootstrap_n=int(args.bootstrap_n),
                    bootstrap_seed=int(args.bootstrap_seed),
                    ci=float(args.ci),
                    rows_csv_path=str(results_dir / f"why_fetch_{task}_rows_{safe}.csv"),
                    summary_csv_path=str(results_dir / f"why_fetch_{task}_summary_{safe}.csv"),
                    smoke=False,
                )
                commands.append(cmd)
                _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "sae_feature_families", False)):
        sae_repo = str(getattr(args, "sae_repo", "")).strip()
        if not sae_repo:
            raise ValueError("--sae_feature_families requires --sae_repo")
        for m in [str(x) for x in models]:
            safe = str(m).replace("/", "_")
            cmd = _feature_families_cmd(
                model_name_or_path=str(m),
                sae_repo=sae_repo,
                sae_layer=int(getattr(args, "sae_layer", 15)),
                sae_width=str(getattr(args, "sae_width", "16k")),
                sae_scale=float(getattr(args, "sae_scale", 1.0)),
                n_features=int(getattr(args, "sae_n_features", 64)),
                max_pairs=int(getattr(args, "sae_max_pairs", 32)),
                similarity_threshold=float(getattr(args, "sae_similarity_threshold", 0.8)),
                null_n=int(getattr(args, "sae_null_n", 500)),
                disamb_path=str(disamb_path),
                device="cuda",
                torch_dtype="bfloat16",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                bootstrap_seed=int(args.bootstrap_seed),
                ci=float(args.ci),
                out_prefix=str(results_dir / f"feature_families_{safe}"),
                smoke=False,
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    if bool(getattr(args, "completeness", False)):
        comp_tasks = str(getattr(args, "completeness_tasks", "disamb,cf,coh"))
        comp_layers = str(getattr(args, "completeness_layers", ""))
        comp_expl = str(getattr(args, "completeness_explanation_path", ""))
        for m in [str(x) for x in models]:
            safe = str(m).replace("/", "_")
            comp_cmd = _completeness_cmd(
                model_name_or_path=str(m),
                tasks=comp_tasks,
                disamb_path=str(disamb_path),
                cf_path=str(cf_path),
                coh_path=str(coh_path),
                layers=comp_layers,
                explanation_path=comp_expl,
                device="cuda",
                torch_dtype="bfloat16",
                attn_implementation="eager",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                bootstrap_n=int(args.bootstrap_n),
                bootstrap_seed=int(args.bootstrap_seed),
                ci=float(args.ci),
                csv_path=str(results_dir / f"completeness_{safe}.csv"),
                smoke=False,
            )
            commands.append(comp_cmd)
            _run(comp_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    report_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "report_results.py"),
        "--results_dir",
        str(results_dir),
        "--out_path",
        str(results_dir / "results_report.md"),
    ]
    commands.append(report_cmd)
    _run(report_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    run = PaperRun(mode="cuda_validated", results_dir=results_dir, dataset_manifest_path=dataset_manifest_path, commands=commands)
    if not bool(args.dry_run):
        _write_run_manifest(run)
    return run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run paper-style evaluations in three reproducible modes.")
    p.add_argument("mode", type=str, choices=["smoke", "m1max", "m1max_safe", "cuda_validated"])
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data_paper_hardened_v2"))
    p.add_argument("--results_dir", type=str, default="")
    p.add_argument("--dataset_seed", type=int, default=0)

    p.add_argument("--local_files_only", action="store_true", help="Disallow model downloads (offline mode).")
    p.add_argument("--trust_remote_code", action="store_true", help="Allow custom HF model code (use with care).")
    p.add_argument("--revision", type=str, default=None, help="Optional HF model revision (branch/tag/commit SHA).")
    p.add_argument(
        "--tokenizer_revision",
        type=str,
        default=None,
        help="Optional HF tokenizer revision (defaults to --revision when unset).",
    )
    p.add_argument("--device_map", type=str, default=None, help="Device map for multi-GPU (behavioral runs only).")

    p.add_argument("--bootstrap_n", type=int, default=1000)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument(
        "--model_timeout_seconds",
        type=int,
        default=1200,
        help="Per-model timeout for the MPS-safe behavioral stage (default: 1200).",
    )

    p.add_argument("--attn_behavioral", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--specificity_depth_frac", type=float, default=0.25)
    p.add_argument("--specificity_buffer", type=int, default=2)
    p.add_argument("--specificity_position_window", type=int, default=8)
    p.add_argument(
        "--mechanistic_disamb_path_decomp",
        action="store_true",
        help="Run DISAMB logit-diff path decomposition stage.",
    )
    p.add_argument(
        "--mechanistic_disamb_max_pairs",
        type=int,
        default=0,
        help="Max DISAMB pairs for decomposition stage (0 = all).",
    )
    p.add_argument(
        "--mechanistic_disamb_mode",
        type=str,
        default="layer",
        choices=["layer", "head"],
        help="Path decomposition mode.",
    )
    p.add_argument(
        "--mechanistic_disamb_position",
        type=int,
        default=-1,
        help="Token position for decomposition readout.",
    )
    p.add_argument(
        "--mechanistic_why_fetch",
        action="store_true",
        help="Run why-fetch QK/OV stage.",
    )
    p.add_argument(
        "--mechanistic_why_fetch_tasks",
        type=str,
        default="disamb",
        help="Comma-separated tasks for why-fetch stage (disamb,cf,coh).",
    )
    p.add_argument(
        "--mechanistic_why_fetch_n_examples",
        type=int,
        default=100,
        help="Max examples per task for why-fetch stage.",
    )
    p.add_argument(
        "--mechanistic_why_fetch_heads_topk",
        type=int,
        default=4,
        help="Top-k heads per layer by fetch mass (0 = all heads).",
    )
    p.add_argument("--sae_feature_families", action="store_true", help="Run SAE feature-family clustering stage.")
    p.add_argument("--sae_repo", type=str, default="", help="SAE repo/path for feature-family stage.")
    p.add_argument("--sae_layer", type=int, default=15)
    p.add_argument("--sae_width", type=str, default="16k")
    p.add_argument("--sae_scale", type=float, default=1.0)
    p.add_argument("--sae_n_features", type=int, default=64)
    p.add_argument("--sae_max_pairs", type=int, default=32)
    p.add_argument("--sae_similarity_threshold", type=float, default=0.8)
    p.add_argument("--sae_null_n", type=int, default=500)
    p.add_argument("--completeness", action="store_true", help="Run explanatory completeness stage.")
    p.add_argument(
        "--completeness_tasks",
        type=str,
        default="disamb,cf,coh",
        help="Comma-separated tasks for completeness stage.",
    )
    p.add_argument(
        "--completeness_layers",
        type=str,
        default="",
        help="Optional comma-separated total-effect layer subset for completeness stage.",
    )
    p.add_argument(
        "--completeness_explanation_path",
        type=str,
        default="",
        help="Optional YAML/JSON explanation config for completeness stage.",
    )
    p.add_argument(
        "--run_clt_patching",
        action="store_true",
        help="Run CLT CPT patching in the primary AoM eval run for each mode.",
    )
    p.add_argument(
        "--run_clt_stage",
        action="store_true",
        help="Run a dedicated DISAMB-only CLT CPT stage and write `clt_cpt_disamb_only.csv`.",
    )
    p.add_argument("--clt_repo", type=str, default="", help="CLT repo id or local bundle path.")
    p.add_argument("--clt_width", type=str, default="16k", help="CLT width (directory tag).")
    p.add_argument("--clt_run_name", type=str, default=None, help="Explicit CLT run subdir name.")
    p.add_argument("--clt_l0_target", type=int, default=None, help="Select CLT run by average_l0_* tag.")
    p.add_argument("--clt_layers", type=str, default="", help="Comma-separated CLT layers to run.")
    p.add_argument("--clt_scale", type=float, default=1.0, help="CLT input scale factor.")
    p.add_argument("--clt_dtype", type=str, default="float32", help="CLT weights/compute dtype.")
    p.add_argument(
        "--clt_decode_strategy",
        type=str,
        default="delta_1decode",
        choices=["safe_2decode", "delta_1decode"],
        help="Decode strategy for CLT latent patching.",
    )
    p.add_argument(
        "--clt_dtype_policy",
        type=str,
        default="clt",
        choices=["clt", "model"],
        help="Dtype policy for CLT hook compute.",
    )
    p.add_argument("--clt_eps_active", type=float, default=1e-6, help="Threshold for active CLT latent counts.")
    p.add_argument(
        "--clt_topk_recovery",
        action="store_true",
        help="Run CLT top-k recovery stage (`aom_clt_topk_recovery.py`).",
    )
    p.add_argument("--clt_topk_layers", type=str, default="4,8,12", help="Comma-separated layers for top-k recovery.")
    p.add_argument(
        "--clt_topk_ks",
        type=str,
        default="1,5,10,20,50,100,200,500,1000,2000,4000,8000,16384",
        help="Comma-separated k grid for top-k recovery.",
    )
    p.add_argument(
        "--clt_topk_logz_ks",
        type=str,
        default="20,50,200,16384",
        help="Comma-separated ks for optional logZ decomposition in top-k recovery.",
    )
    p.add_argument("--clt_topk_with_logz", action="store_true", help="Enable optional logZ decomposition in top-k recovery.")
    p.add_argument("--clt_topk_split_seed", type=int, default=0)
    p.add_argument("--clt_topk_frac_selection", type=float, default=0.5)
    p.add_argument("--clt_topk_random_k_seeds", type=str, default="0,1,2,3,4")
    p.add_argument(
        "--clt_topk_random_control_mode",
        type=str,
        default="complement",
        choices=["complement", "matched_bin"],
    )
    p.add_argument("--clt_topk_matched_bin_n_bins", type=int, default=10)
    p.add_argument("--clt_topk_eps", type=float, default=1e-6)
    p.add_argument(
        "--clt_feature_analysis",
        action="store_true",
        help="Run CLT top-feature activation/selectivity/family analysis stage.",
    )
    p.add_argument("--clt_feature_layer", type=int, default=4)
    p.add_argument("--clt_feature_top_n_features", type=int, default=50)
    p.add_argument("--clt_feature_max_pairs", type=int, default=0)
    p.add_argument("--clt_feature_similarity_threshold", type=float, default=0.8)
    p.add_argument("--clt_feature_stability_bootstrap_n", type=int, default=200)
    p.add_argument("--clt_feature_null_n", type=int, default=500)
    p.add_argument(
        "--clt_head_attribution",
        action="store_true",
        help="Run CLT feature-space head attribution + top-vs-random ablation stage.",
    )
    p.add_argument("--clt_head_layers", type=str, default="1,2,3")
    p.add_argument("--clt_head_top_h", type=int, default=5)
    p.add_argument("--clt_head_max_pairs", type=int, default=0)
    p.add_argument("--clt_head_ablate_positions", type=str, default="target", choices=["target", "all"])
    p.add_argument("--clt_head_qk_patterns", action="store_true")
    p.add_argument(
        "--specificity_selection_seeds",
        type=str,
        default="0",
        help="Comma-separated selection seeds for specificity control off-target pair selection (default: 0).",
    )

    p.add_argument("--skip_dataset", action="store_true", help="Skip dataset generation/manifest writing.")
    p.add_argument("--skip_patching", action="store_true", help="Skip CPT layer-sweep patching runs.")
    p.add_argument("--skip_specificity", action="store_true", help="Skip CPT target-specificity runs.")
    p.add_argument("--skip_cf_patching", action="store_true", help="Skip AoM-CF causal patching runs.")
    p.add_argument("--skip_coh_patching", action="store_true", help="Skip AoM-COH causal patching runs.")
    p.add_argument("--dry_run", action="store_true", help="Print commands without running.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results_dir:
        args.results_dir = str(ROOT / "results" / f"paper_{args.mode}")

    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "m1max":
        run_m1max(args)
    elif args.mode == "m1max_safe":
        run_m1max_safe(args)
    elif args.mode == "cuda_validated":
        run_a100(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode!r}")


if __name__ == "__main__":
    main()
