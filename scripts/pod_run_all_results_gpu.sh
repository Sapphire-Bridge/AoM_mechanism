#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pod_env.sh"

RESULTS_DIR="${1:-$REPO_ROOT/results/paper_cuda_validated}"
LOG_PATH="${RESULTS_DIR}.log"
PID_PATH="${RESULTS_DIR}.pid"
STATUS_PATH="${RESULTS_DIR}.status"
ARCHIVE_PATH="$(pod_archive_path "$RESULTS_DIR")"

mkdir -p "$(dirname "$RESULTS_DIR")"
pod_require_empty_dir "$RESULTS_DIR"

cmd=(
  "$REPO_ROOT/scripts/pod_after_success.sh"
  "--result-path"
  "$RESULTS_DIR"
  "--archive-path"
  "$ARCHIVE_PATH"
  "--status-path"
  "$STATUS_PATH"
  "--pid-path"
  "$PID_PATH"
  "--"
  "$REPO_ROOT/.venv/bin/python"
  "scripts/run_paper_accelerated.py"
  "--results_dir"
  "$RESULTS_DIR"
)
if pod_bool_enabled "${LOCAL_FILES_ONLY:-0}"; then
  cmd+=("--local_files_only")
fi

echo "results_dir: $RESULTS_DIR"
echo "archive: $ARCHIVE_PATH"
echo "status_file: $STATUS_PATH"
pod_launch_background "$LOG_PATH" "$PID_PATH" "${cmd[@]}"
