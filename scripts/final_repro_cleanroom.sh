#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="m1max"
GIT_REV="HEAD"
CLEAN_DIR=""
PYTHON_BIN="python3"
USE_VENV=1
VENV_NAME=".venv"
INSTALL_DEPS=1
LOCAL_FILES_ONLY=0
HF_REVISION=""
HF_TOKENIZER_REVISION=""
RESULTS_REL="results_submission"
TABLES_REL="tables_submission"
ARCHIVE_REL="aom_replication_bundle.tar.gz"
ALLOW_DIRTY=0
SKIP_RUN=0
SKIP_TABLES=0
SKIP_CHECKS=0
COPY_BACK_DIR=""

usage() {
  cat <<'EOF'
Run an end-to-end clean-room reproducibility test from a pinned git commit.

Workflow:
1) Export a fresh repo snapshot from --git-rev into --clean-dir.
2) (Optional) create venv + install requirements.
3) Run scripts/build_replication_bundle.sh in the clean repo.
4) (Optional) run evidence-contract checks/tests in the clean repo.

Usage:
  bash scripts/final_repro_cleanroom.sh [options]

Options:
  --mode MODE                  run_paper mode: smoke | m1max | cuda_validated (default: m1max)
  --git-rev REV                git revision to export (default: HEAD)
  --clean-dir PATH             clean-room directory (default: /tmp/aom_cleanroom_<utc>)
  --python BIN                 python executable (default: python3)
  --no-venv                    do not create/use virtualenv
  --venv-name NAME             venv directory name inside clean repo (default: .venv)
  --no-install                 skip pip install -r requirements.txt
  --local-files-only           pass --local_files_only to run_paper
  --revision REV               HF model revision pin passed through to run_paper
  --tokenizer-revision REV     HF tokenizer revision pin passed through to run_paper
  --results-rel PATH           results directory (repo-relative in clean repo)
  --tables-rel PATH            tables directory (repo-relative in clean repo)
  --archive-rel PATH           replication archive path (repo-relative in clean repo)
  --allow-dirty                allow exporting from a dirty source worktree
  --skip-run                   pass --skip-run to build_replication_bundle.sh
  --skip-tables                pass --skip-tables to build_replication_bundle.sh
  --skip-checks                skip evidence-contract checks after the run
  --copy-back-dir PATH         copy final archive + sha256 to this directory
  -h, --help                   show this help

Examples:
  bash scripts/final_repro_cleanroom.sh \
    --mode m1max \
    --git-rev HEAD \
    --revision <hf_commit> \
    --tokenizer-revision <hf_commit>

  bash scripts/final_repro_cleanroom.sh \
    --mode cuda_validated \
    --clean-dir /tmp/aom_final_release \
    --copy-back-dir release_artifacts
EOF
}

die() {
  echo "[error] $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "Missing value for --mode"
      MODE="$2"
      shift 2
      ;;
    --git-rev)
      [[ $# -ge 2 ]] || die "Missing value for --git-rev"
      GIT_REV="$2"
      shift 2
      ;;
    --clean-dir)
      [[ $# -ge 2 ]] || die "Missing value for --clean-dir"
      CLEAN_DIR="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "Missing value for --python"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --no-venv)
      USE_VENV=0
      shift
      ;;
    --venv-name)
      [[ $# -ge 2 ]] || die "Missing value for --venv-name"
      VENV_NAME="$2"
      shift 2
      ;;
    --no-install)
      INSTALL_DEPS=0
      shift
      ;;
    --local-files-only)
      LOCAL_FILES_ONLY=1
      shift
      ;;
    --revision)
      [[ $# -ge 2 ]] || die "Missing value for --revision"
      HF_REVISION="$2"
      shift 2
      ;;
    --tokenizer-revision)
      [[ $# -ge 2 ]] || die "Missing value for --tokenizer-revision"
      HF_TOKENIZER_REVISION="$2"
      shift 2
      ;;
    --results-rel)
      [[ $# -ge 2 ]] || die "Missing value for --results-rel"
      RESULTS_REL="$2"
      shift 2
      ;;
    --tables-rel)
      [[ $# -ge 2 ]] || die "Missing value for --tables-rel"
      TABLES_REL="$2"
      shift 2
      ;;
    --archive-rel)
      [[ $# -ge 2 ]] || die "Missing value for --archive-rel"
      ARCHIVE_REL="$2"
      shift 2
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
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
    --skip-checks)
      SKIP_CHECKS=1
      shift
      ;;
    --copy-back-dir)
      [[ $# -ge 2 ]] || die "Missing value for --copy-back-dir"
      COPY_BACK_DIR="$2"
      shift 2
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
  smoke|m1max|cuda_validated) ;;
  *) die "Invalid --mode: $MODE (expected smoke|m1max|cuda_validated)" ;;
esac

need_cmd git
need_cmd tar
need_cmd "$PYTHON_BIN"

if [[ -z "$CLEAN_DIR" ]]; then
  CLEAN_DIR="/tmp/aom_cleanroom_$(date -u +%Y%m%d_%H%M%S)"
fi

SOURCE_STATUS="$(git -C "$ROOT" status --porcelain || true)"
if [[ "$ALLOW_DIRTY" -ne 1 && -n "$SOURCE_STATUS" ]]; then
  die "Source worktree is dirty. Commit/stash first, or pass --allow-dirty."
fi

git -C "$ROOT" rev-parse --verify "$GIT_REV^{commit}" >/dev/null 2>&1 || die "Unknown git revision: $GIT_REV"
RESOLVED_REV="$(git -C "$ROOT" rev-parse "$GIT_REV")"

mkdir -p "$CLEAN_DIR"
if [[ -n "$(ls -A "$CLEAN_DIR" 2>/dev/null || true)" ]]; then
  die "--clean-dir is not empty: $CLEAN_DIR"
fi

CLEAN_REPO="$CLEAN_DIR/repo"
mkdir -p "$CLEAN_REPO"

echo "[info] exporting commit $GIT_REV to $CLEAN_REPO"
git -C "$ROOT" archive --format=tar "$GIT_REV" | tar -xf - -C "$CLEAN_REPO"
echo "$RESOLVED_REV" > "$CLEAN_REPO/SOURCE_GIT_COMMIT.txt"

[[ -f "$CLEAN_REPO/scripts/build_replication_bundle.sh" ]] || die "Clean export missing scripts/build_replication_bundle.sh"
[[ -f "$CLEAN_REPO/requirements.txt" ]] || die "Clean export missing requirements.txt"

PY_RUN="$PYTHON_BIN"
if [[ "$USE_VENV" -eq 1 ]]; then
  echo "[run] $PYTHON_BIN -m venv $CLEAN_REPO/$VENV_NAME"
  "$PYTHON_BIN" -m venv "$CLEAN_REPO/$VENV_NAME"
  PY_RUN="$CLEAN_REPO/$VENV_NAME/bin/python"
fi

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  REQ_FILE="$CLEAN_REPO/requirements.txt"
  if [[ -f "$CLEAN_REPO/requirements.lock.txt" ]]; then
    REQ_FILE="$CLEAN_REPO/requirements.lock.txt"
  fi
  echo "[run] $PY_RUN -m pip install -r $(basename "$REQ_FILE")"
  "$PY_RUN" -m pip install -r "$REQ_FILE"
fi

bundle_cmd=(
  bash
  scripts/build_replication_bundle.sh
  --mode
  "$MODE"
  --results-dir
  "$RESULTS_REL"
  --tables-dir
  "$TABLES_REL"
  --archive
  "$ARCHIVE_REL"
)

if [[ "$LOCAL_FILES_ONLY" -eq 1 ]]; then
  bundle_cmd+=(--local-files-only)
fi
if [[ -n "$HF_REVISION" ]]; then
  bundle_cmd+=(--revision "$HF_REVISION")
fi
if [[ -n "$HF_TOKENIZER_REVISION" ]]; then
  bundle_cmd+=(--tokenizer-revision "$HF_TOKENIZER_REVISION")
fi
if [[ "$SKIP_RUN" -eq 1 ]]; then
  bundle_cmd+=(--skip-run)
fi
if [[ "$SKIP_TABLES" -eq 1 ]]; then
  bundle_cmd+=(--skip-tables)
fi

echo "[run] ${bundle_cmd[*]}"
if [[ "$USE_VENV" -eq 1 ]]; then
  (cd "$CLEAN_REPO" && PATH="$CLEAN_REPO/$VENV_NAME/bin:$PATH" "${bundle_cmd[@]}")
else
  (cd "$CLEAN_REPO" && "${bundle_cmd[@]}")
fi

if [[ "$SKIP_CHECKS" -eq 0 ]]; then
  echo "[run] $PY_RUN scripts/check_evidence_contract.py"
  (cd "$CLEAN_REPO" && "$PY_RUN" scripts/check_evidence_contract.py)
  echo "[run] $PY_RUN scripts/check_evidence_contract_fields.py"
  (cd "$CLEAN_REPO" && "$PY_RUN" scripts/check_evidence_contract_fields.py)
  echo "[run] $PY_RUN -m pytest -q tests/test_evidence_contract_ids.py tests/test_evidence_contract_fields.py"
  (cd "$CLEAN_REPO" && "$PY_RUN" -m pytest -q tests/test_evidence_contract_ids.py tests/test_evidence_contract_fields.py)
fi

ARCHIVE_ABS="$CLEAN_REPO/$ARCHIVE_REL"
SHA_ABS="${ARCHIVE_ABS}.sha256"
[[ -f "$ARCHIVE_ABS" ]] || die "Expected archive not found: $ARCHIVE_ABS"
[[ -f "$SHA_ABS" ]] || die "Expected SHA file not found: $SHA_ABS"

if [[ -n "$COPY_BACK_DIR" ]]; then
  mkdir -p "$COPY_BACK_DIR"
  cp "$ARCHIVE_ABS" "$COPY_BACK_DIR/"
  cp "$SHA_ABS" "$COPY_BACK_DIR/"
  echo "[ok] copied archive artifacts to $COPY_BACK_DIR"
fi

echo "[ok] clean-room reproducibility run complete"
echo "[ok] clean repo:      $CLEAN_REPO"
echo "[ok] results dir:     $CLEAN_REPO/$RESULTS_REL"
echo "[ok] tables dir:      $CLEAN_REPO/$TABLES_REL"
echo "[ok] archive:         $ARCHIVE_ABS"
echo "[ok] archive sha256:  $SHA_ABS"
