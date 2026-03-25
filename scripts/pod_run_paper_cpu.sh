#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pod_env.sh"

RUN_ROOT="${1:-$REPO_ROOT/results/mom_paper_cpu}"
LOG_PATH="${RUN_ROOT}.log"
PID_PATH="${RUN_ROOT}.pid"
STATUS_PATH="${RUN_ROOT}.status"
ARCHIVE_PATH="$(pod_archive_path "$RUN_ROOT")"

mkdir -p "$(dirname "$RUN_ROOT")"
pod_require_empty_dir "$RUN_ROOT"

cmd=(
  "$REPO_ROOT/scripts/pod_after_success.sh"
  "--result-path"
  "$RUN_ROOT"
  "--archive-path"
  "$ARCHIVE_PATH"
  "--status-path"
  "$STATUS_PATH"
  "--pid-path"
  "$PID_PATH"
  "--"
  "$REPO_ROOT/.venv/bin/python"
  "scripts/run_mom_paper.py"
  "--run_root"
  "$RUN_ROOT"
)
if pod_bool_enabled "${LOCAL_FILES_ONLY:-0}"; then
  cmd+=("--local_files_only")
fi

echo "run_root: $RUN_ROOT"
echo "archive: $ARCHIVE_PATH"
echo "status_file: $STATUS_PATH"
pod_launch_background "$LOG_PATH" "$PID_PATH" "${cmd[@]}"
