#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="m1max"
RESULTS_DIR="$ROOT/results_submission"
TABLES_DIR="$ROOT/tables_submission"
ARCHIVE="$ROOT/aom_replication_bundle.tar.gz"
REVISION=""
TOKENIZER_REVISION=""
LOCAL_FILES_ONLY=0
SKIP_RUN=0
SKIP_TABLES=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Build a paper-ready AoM replication bundle.

Usage:
  bash scripts/build_replication_bundle.sh [options]

Options:
  --mode MODE                  run_paper mode: smoke | m1max | a100 (default: m1max)
  --results-dir PATH           run/results directory (default: results_submission)
  --tables-dir PATH            output directory for strict LaTeX tables (default: tables_submission)
  --archive PATH               output archive path (default: aom_replication_bundle.tar.gz)
  --revision REV               HF model revision pin passed to run_paper
  --tokenizer-revision REV     HF tokenizer revision pin passed to run_paper
  --local-files-only           pass --local_files_only to run_paper
  --skip-run                   do not run scripts/run_paper.py
  --skip-tables                do not run strict table regeneration
  --dry-run, --dry_run         print planned actions without running or packaging
  -h, --help                   show this help

Examples:
  bash scripts/build_replication_bundle.sh --mode m1max --revision <hf_commit>
  bash scripts/build_replication_bundle.sh --skip-run --results-dir results/paper_m1max --tables-dir tables_submission
EOF
}

die() {
  echo "[error] $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "Missing required file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing required directory: $1"
}

abs_under_root_to_rel() {
  local abs="$1"
  case "$abs" in
    "$ROOT"/*)
      echo "${abs#"$ROOT"/}"
      ;;
    *)
      die "Path must be inside repo root ($ROOT): $abs"
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "Missing value for --mode"
      MODE="$2"
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
    --archive)
      [[ $# -ge 2 ]] || die "Missing value for --archive"
      ARCHIVE="$2"
      shift 2
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
    --local-files-only)
      LOCAL_FILES_ONLY=1
      shift
      ;;
    --skip-run)
      SKIP_RUN=1
      shift
      ;;
    --skip-tables)
      SKIP_TABLES=1
      shift
      ;;
    --dry-run|--dry_run)
      DRY_RUN=1
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

case "$MODE" in
  smoke|m1max|a100) ;;
  *) die "Invalid --mode: $MODE (expected smoke|m1max|a100)" ;;
esac

RESULTS_DIR_ABS="$(cd "$(dirname "$RESULTS_DIR")" && pwd)/$(basename "$RESULTS_DIR")"
TABLES_DIR_ABS="$(cd "$(dirname "$TABLES_DIR")" && pwd)/$(basename "$TABLES_DIR")"
ARCHIVE_ABS="$(cd "$(dirname "$ARCHIVE")" && pwd)/$(basename "$ARCHIVE")"

if [[ "$SKIP_RUN" -eq 0 ]]; then
  mkdir -p "$RESULTS_DIR_ABS"
  cmd=(
    python
    "$ROOT/scripts/run_paper.py"
    "$MODE"
    --results_dir
    "$RESULTS_DIR_ABS"
  )
  if [[ "$LOCAL_FILES_ONLY" -eq 1 ]]; then
    cmd+=(--local_files_only)
  fi
  if [[ -n "$REVISION" ]]; then
    cmd+=(--revision "$REVISION")
  fi
  if [[ -n "$TOKENIZER_REVISION" ]]; then
    cmd+=(--tokenizer_revision "$TOKENIZER_REVISION")
  fi
  echo "[run] ${cmd[*]}"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${cmd[@]}"
  fi
fi

if [[ "$SKIP_TABLES" -eq 0 ]]; then
  mkdir -p "$TABLES_DIR_ABS"
  echo "[run] MAKE_TABLES_STRICT=1 make tables RESULTS_DIR=$RESULTS_DIR_ABS TABLES_OUT_DIR=$TABLES_DIR_ABS"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    MAKE_TABLES_STRICT=1 make tables RESULTS_DIR="$RESULTS_DIR_ABS" TABLES_OUT_DIR="$TABLES_DIR_ABS"
  fi
fi

require_dir "$ROOT/data_paper_hardened_v2"
require_file "$ROOT/data_paper_hardened_v2/disamb_pairs.jsonl"
require_file "$ROOT/data_paper_hardened_v2/counterfactual.jsonl"
require_file "$ROOT/data_paper_hardened_v2/coherence.jsonl"
require_file "$ROOT/data_paper_hardened_v2/DATASET_MANIFEST.json"

if [[ "$DRY_RUN" -eq 0 ]]; then
  require_dir "$RESULTS_DIR_ABS"
  require_file "$RESULTS_DIR_ABS/RUN_MANIFEST.json"
  require_file "$RESULTS_DIR_ABS/results_report.md"
elif [[ ! -d "$RESULTS_DIR_ABS" ]]; then
  echo "[warn] Dry run: results directory does not exist yet: $RESULTS_DIR_ABS"
fi

results_csv=()
results_manifests=()
if [[ -d "$RESULTS_DIR_ABS" ]]; then
  shopt -s nullglob
  results_csv=("$RESULTS_DIR_ABS"/*.csv)
  results_manifests=("$RESULTS_DIR_ABS"/*.manifest.json)
  shopt -u nullglob
fi
if [[ "$DRY_RUN" -eq 0 ]]; then
  (( ${#results_csv[@]} > 0 )) || die "No CSV files found in results directory: $RESULTS_DIR_ABS"
  (( ${#results_manifests[@]} > 0 )) || die "No .manifest.json files found in results directory: $RESULTS_DIR_ABS"
fi

if [[ "$SKIP_TABLES" -eq 0 ]]; then
  if [[ "$DRY_RUN" -eq 0 ]]; then
    require_dir "$TABLES_DIR_ABS"
    require_file "$TABLES_DIR_ABS/aom_eval.tex"
    require_file "$TABLES_DIR_ABS/cf_patching.tex"
    require_file "$TABLES_DIR_ABS/coh_patching.tex"
  elif [[ ! -d "$TABLES_DIR_ABS" ]]; then
    echo "[warn] Dry run: tables directory does not exist yet: $TABLES_DIR_ABS"
  fi
fi

results_rel="$(abs_under_root_to_rel "$RESULTS_DIR_ABS")"
tables_rel=""
if [[ -d "$TABLES_DIR_ABS" ]]; then
  tables_rel="$(abs_under_root_to_rel "$TABLES_DIR_ABS")"
fi

bundle_paths=(
  "LICENSE"
  "CITATION.cff"
  "README.md"
  "paper/MoM_paper.md"
  "MoM_evidence_contract.md"
  "PAPER_MODES.md"
  "PAPER_VERIFICATION_GUIDE.md"
  "Makefile"
  "requirements.txt"
  "requirements.lock.txt"
  "aom"
  "aom_eval.py"
  "aom_cf_patching.py"
  "aom_coh_patching.py"
  "scripts/run_paper.py"
  "scripts/report_results.py"
  "scripts/make_tables.py"
  "scripts/build_replication_bundle.sh"
  "tables/table_aom_eval.py"
  "tables/table_cf_patching.py"
  "tables/table_coh_patching.py"
  "data/README.md"
  "data_paper_hardened_v2"
  "$results_rel"
)

if [[ -f "$ROOT/SOURCE_GIT_COMMIT.txt" ]]; then
  bundle_paths+=("SOURCE_GIT_COMMIT.txt")
fi

optional_bundle_paths=(
  ".zenodo.json"
  "docs/ARCHIVAL_RELEASE_MODEL.md"
  "docs/PUBLIC_ARTIFACT_FORMAT.md"
  "docs/RELEASE_CHECKLIST.md"
  "reports/archive_readiness_check.md"
  "reports/readme_reproduction_report.md"
  "reports/readme_reproduction_log.json"
  "reports/archival_readiness_final.md"
)

for rel in "${optional_bundle_paths[@]}"; do
  if [[ -e "$ROOT/$rel" ]]; then
    bundle_paths+=("$rel")
  fi
done

if [[ -n "$tables_rel" ]]; then
  bundle_paths+=("$tables_rel")
fi

mkdir -p "$(dirname "$ARCHIVE_ABS")"
echo "[run] tar -czf $ARCHIVE_ABS ..."
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%s\n' "${bundle_paths[@]}"
  echo "[info] dry run only; archive not written"
  exit 0
fi
tar \
  --exclude="*/__pycache__/*" \
  --exclude="*.pyc" \
  --exclude=".DS_Store" \
  -czf "$ARCHIVE_ABS" \
  -C "$ROOT" \
  "${bundle_paths[@]}"

checksum_path="${ARCHIVE_ABS}.sha256"
if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$ARCHIVE_ABS" > "$checksum_path"
elif command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$ARCHIVE_ABS" > "$checksum_path"
else
  die "No SHA-256 tool found (expected shasum or sha256sum)"
fi

echo "[ok] wrote archive: $ARCHIVE_ABS"
echo "[ok] wrote digest:  $checksum_path"
