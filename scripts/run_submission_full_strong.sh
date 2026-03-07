#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="$ROOT/data_paper_hardened_v2"
RESULTS_DIR="$ROOT/results_submission_full"
TABLES_DIR="$ROOT/tables_submission_full"

CORE_DEVICE="cpu"
CORE_TORCH_DTYPE="float32"
SPEC_DEVICE="auto"
SPEC_TORCH_DTYPE="float16"

BOOTSTRAP_N=1000
BOOTSTRAP_SEED=42
CI=0.95
SPEC_DEPTH_FRAC=0.25
SPEC_BUFFER=2
SPEC_WINDOW=8
SPEC_SEED=0

LOCAL_FILES_ONLY=0
REVISION=""
TOKENIZER_REVISION=""
SKIP_TABLES=0

usage() {
  cat <<'EOF'
Run a full hardened submission suite (behavioral + CPT + specificity + CF/COH patching).

Outputs:
  <results-dir>/aom_eval.csv
  <results-dir>/cpt_specificity_disamb_only.csv
  <results-dir>/cf_patching.csv
  <results-dir>/coh_patching.csv
  <results-dir>/results_report.md
  <results-dir>/RUN_MANIFEST.json
  <tables-dir>/*.tex (unless --skip-tables)

Usage:
  bash scripts/run_submission_full_strong.sh [options]

Options:
  --data-dir PATH             dataset directory with DATASET_MANIFEST.json
  --results-dir PATH          output results directory
  --tables-dir PATH           output tables directory
  --core-device DEV           device for AoM+CPT+CF/COH (default: cpu)
  --core-torch-dtype DTYPE    torch dtype for AoM+CPT (default: float32)
  --spec-device DEV           device for specificity run (default: auto)
  --spec-torch-dtype DTYPE    torch dtype for specificity (default: float16)
  --bootstrap-n N             bootstrap replicates (default: 1000)
  --bootstrap-seed N          bootstrap seed (default: 42)
  --ci FLOAT                  CI level (default: 0.95)
  --spec-depth-frac FLOAT     specificity depth frac (default: 0.25)
  --spec-buffer N             specificity exclusion buffer (default: 2)
  --spec-window N             specificity position window (default: 8)
  --spec-seed N               specificity selection seed (default: 0)
  --local-files-only          pass --local_files_only to all model runs
  --revision REV              HF model revision
  --tokenizer-revision REV    HF tokenizer revision
  --skip-tables               skip strict table regeneration
  -h, --help                  show this help
EOF
}

die() {
  echo "[error] $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      [[ $# -ge 2 ]] || die "Missing value for --data-dir"
      DATA_DIR="$2"
      shift 2
      ;;
    --results-dir)
      [[ $# -ge 2 ]] || die "Missing value for --results-dir"
      RESULTS_DIR="$2"
      shift 2
      ;;
    --tables-dir)
      [[ $# -ge 2 ]] || die "Missing value for --tables-dir"
      TABLES_DIR="$2"
      shift 2
      ;;
    --core-device)
      [[ $# -ge 2 ]] || die "Missing value for --core-device"
      CORE_DEVICE="$2"
      shift 2
      ;;
    --core-torch-dtype)
      [[ $# -ge 2 ]] || die "Missing value for --core-torch-dtype"
      CORE_TORCH_DTYPE="$2"
      shift 2
      ;;
    --spec-device)
      [[ $# -ge 2 ]] || die "Missing value for --spec-device"
      SPEC_DEVICE="$2"
      shift 2
      ;;
    --spec-torch-dtype)
      [[ $# -ge 2 ]] || die "Missing value for --spec-torch-dtype"
      SPEC_TORCH_DTYPE="$2"
      shift 2
      ;;
    --bootstrap-n)
      [[ $# -ge 2 ]] || die "Missing value for --bootstrap-n"
      BOOTSTRAP_N="$2"
      shift 2
      ;;
    --bootstrap-seed)
      [[ $# -ge 2 ]] || die "Missing value for --bootstrap-seed"
      BOOTSTRAP_SEED="$2"
      shift 2
      ;;
    --ci)
      [[ $# -ge 2 ]] || die "Missing value for --ci"
      CI="$2"
      shift 2
      ;;
    --spec-depth-frac)
      [[ $# -ge 2 ]] || die "Missing value for --spec-depth-frac"
      SPEC_DEPTH_FRAC="$2"
      shift 2
      ;;
    --spec-buffer)
      [[ $# -ge 2 ]] || die "Missing value for --spec-buffer"
      SPEC_BUFFER="$2"
      shift 2
      ;;
    --spec-window)
      [[ $# -ge 2 ]] || die "Missing value for --spec-window"
      SPEC_WINDOW="$2"
      shift 2
      ;;
    --spec-seed)
      [[ $# -ge 2 ]] || die "Missing value for --spec-seed"
      SPEC_SEED="$2"
      shift 2
      ;;
    --local-files-only)
      LOCAL_FILES_ONLY=1
      shift
      ;;
    --revision)
      [[ $# -ge 2 ]] || die "Missing value for --revision"
      REVISION="$2"
      shift 2
      ;;
    --tokenizer-revision)
      [[ $# -ge 2 ]] || die "Missing value for --tokenizer-revision"
      TOKENIZER_REVISION="$2"
      shift 2
      ;;
    --skip-tables)
      SKIP_TABLES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

DISAMB_PATH="$DATA_DIR/disamb_pairs.jsonl"
CF_PATH="$DATA_DIR/counterfactual.jsonl"
COH_PATH="$DATA_DIR/coherence.jsonl"
DATASET_MANIFEST_PATH="$DATA_DIR/DATASET_MANIFEST.json"

[[ -f "$DISAMB_PATH" ]] || die "Missing DISAMB dataset: $DISAMB_PATH"
[[ -f "$CF_PATH" ]] || die "Missing CF dataset: $CF_PATH"
[[ -f "$COH_PATH" ]] || die "Missing COH dataset: $COH_PATH"
[[ -f "$DATASET_MANIFEST_PATH" ]] || die "Missing dataset manifest: $DATASET_MANIFEST_PATH"

mkdir -p "$RESULTS_DIR"

cmd_common=(
  --bootstrap_n "$BOOTSTRAP_N"
  --bootstrap_seed "$BOOTSTRAP_SEED"
  --ci "$CI"
  --dataset_manifest_path "$DATASET_MANIFEST_PATH"
  --error_policy raise
  --max_fail_rate 0
  --strict_data
  --no-require_git
)
if [[ "$LOCAL_FILES_ONLY" -eq 1 ]]; then
  cmd_common+=(--local_files_only)
fi
if [[ -n "$REVISION" ]]; then
  cmd_common+=(--revision "$REVISION")
fi
if [[ -n "$TOKENIZER_REVISION" ]]; then
  cmd_common+=(--tokenizer_revision "$TOKENIZER_REVISION")
fi

COMMAND_LOG="$RESULTS_DIR/commands.log"
: > "$COMMAND_LOG"

run_and_log() {
  local -a cmd=("$@")
  echo "[run] ${cmd[*]}"
  printf "%s\n" "${cmd[*]}" >> "$COMMAND_LOG"
  "${cmd[@]}"
}

# 1) Behavioral AoM + CPT layer sweep in one artifact (paper core models).
run_and_log python aom_eval.py \
  --models gpt2 Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B \
  --device "$CORE_DEVICE" \
  --attn_implementation eager \
  --torch_dtype "$CORE_TORCH_DTYPE" \
  --logprobs_dtype float32 \
  --disamb_path "$DISAMB_PATH" \
  --cf_path "$CF_PATH" \
  --coh_path "$COH_PATH" \
  --run_patching \
  --strict_metrics \
  --csv_path "$RESULTS_DIR/aom_eval.csv" \
  "${cmd_common[@]}"

# 2) Fixed-layer CPT specificity across 8 paper models.
run_and_log python aom_eval.py \
  --models \
    gpt2 \
    Qwen/Qwen2.5-0.5B \
    Qwen/Qwen2.5-1.5B \
    Qwen/Qwen2.5-3B \
    Qwen/Qwen3-4B \
    meta-llama/Llama-3.2-1B \
    meta-llama/Llama-3.2-3B \
    meta-llama/Meta-Llama-3.1-8B \
  --device "$SPEC_DEVICE" \
  --attn_implementation eager \
  --torch_dtype "$SPEC_TORCH_DTYPE" \
  --logprobs_dtype float32 \
  --disamb_path "$DISAMB_PATH" \
  --cf_path "" \
  --coh_path "" \
  --run_patching_specificity \
  --patch_specificity_depth_frac "$SPEC_DEPTH_FRAC" \
  --patch_specificity_buffer "$SPEC_BUFFER" \
  --patch_specificity_position_window "$SPEC_WINDOW" \
  --patch_specificity_seed "$SPEC_SEED" \
  --composite_missing_policy ignore \
  --csv_path "$RESULTS_DIR/cpt_specificity_disamb_only.csv" \
  "${cmd_common[@]}"

# 3) CF intervention-span patching (shift/invariant, left_aligned_truncated).
run_and_log python aom_cf_patching.py \
  --models gpt2 Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B \
  --device "$CORE_DEVICE" \
  --attn_implementation eager \
  --cf_path "$CF_PATH" \
  --span_mode left_aligned_truncated \
  --include_expected_effects shift,invariant \
  --sweep_seeds 0 \
  --csv_path "$RESULTS_DIR/cf_patching.csv" \
  "${cmd_common[@]}"

# 4) COH pseudo-ablation patching (constraint vs irrelevant).
run_and_log python aom_coh_patching.py \
  --models gpt2 Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B \
  --device "$CORE_DEVICE" \
  --attn_implementation eager \
  --coh_path "$COH_PATH" \
  --sweep_seeds 0 \
  --csv_path "$RESULTS_DIR/coh_patching.csv" \
  "${cmd_common[@]}"

run_and_log python scripts/report_results.py \
  --results_dir "$RESULTS_DIR" \
  --out_path "$RESULTS_DIR/results_report.md"

if [[ "$SKIP_TABLES" -eq 0 ]]; then
  mkdir -p "$TABLES_DIR"
  run_and_log make tables \
    RESULTS_DIR="$RESULTS_DIR" \
    TABLES_OUT_DIR="$TABLES_DIR" \
    MAKE_TABLES_STRICT=1
fi

SOURCE_GIT_COMMIT=""
if [[ -f "$ROOT/SOURCE_GIT_COMMIT.txt" ]]; then
  SOURCE_GIT_COMMIT="$(cat "$ROOT/SOURCE_GIT_COMMIT.txt")"
elif git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
fi

python - <<'PY' "$RESULTS_DIR" "$DATASET_MANIFEST_PATH" "$SOURCE_GIT_COMMIT" "$COMMAND_LOG"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

results_dir = Path(sys.argv[1])
dataset_manifest = Path(sys.argv[2])
git_commit = sys.argv[3]
command_log = Path(sys.argv[4])
commands = [line.strip() for line in command_log.read_text(encoding="utf-8").splitlines() if line.strip()]
manifest = {
    "mode": "submission_full_strong",
    "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "results_dir": str(results_dir.resolve()),
    "dataset_manifest_path": str(dataset_manifest.resolve()),
    "git_commit": str(git_commit),
    "commands": commands,
}
(results_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "[ok] full submission suite completed"
echo "[ok] results: $RESULTS_DIR"
if [[ "$SKIP_TABLES" -eq 0 ]]; then
  echo "[ok] tables:  $TABLES_DIR"
fi
