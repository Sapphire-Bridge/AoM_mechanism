#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This runs under nohup as a fresh shell process, so it must re-source pod_env.sh.
source "$SCRIPT_DIR/pod_env.sh"

RESULT_PATH=""
ARCHIVE_PATH=""
STATUS_PATH=""
PID_PATH=""
POST_SUCCESS_ACTION="${RUNPOD_POST_SUCCESS_ACTION:-auto}"

usage() {
  cat <<'EOF'
Run a command, archive its result path on success, optionally stop/terminate a RunPod pod,
and write a simple status file.

Usage:
  bash scripts/pod_after_success.sh \
    --result-path PATH \
    --archive-path PATH \
    --status-path PATH \
    --pid-path PATH \
    [--post-success-action auto|none|stop|terminate] \
    -- <command ...>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result-path)
      RESULT_PATH="$2"
      shift 2
      ;;
    --archive-path)
      ARCHIVE_PATH="$2"
      shift 2
      ;;
    --status-path)
      STATUS_PATH="$2"
      shift 2
      ;;
    --pid-path)
      PID_PATH="$2"
      shift 2
      ;;
    --post-success-action)
      POST_SUCCESS_ACTION="$2"
      shift 2
      ;;
    --)
      shift
      break
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

[[ -n "$RESULT_PATH" ]] || die "--result-path is required"
[[ -n "$ARCHIVE_PATH" ]] || die "--archive-path is required"
[[ -n "$STATUS_PATH" ]] || die "--status-path is required"
[[ -n "$PID_PATH" ]] || die "--pid-path is required"
(( $# > 0 )) || die "Missing command after --"

CHECKSUM_PATH="$(pod_checksum_path "$ARCHIVE_PATH")"
pod_status_write "$STATUS_PATH" "RUNNING"

cleanup() {
  rm -f "$PID_PATH"
}
trap cleanup EXIT

if "$@"; then
  :
else
  # In the else branch of `if "$@"`, $? is the wrapped command's real exit code.
  code="$?"
  pod_status_write "$STATUS_PATH" "FAILED: command exit $code"
  exit "$code"
fi

pod_archive_result "$RESULT_PATH" "$ARCHIVE_PATH" "$CHECKSUM_PATH"
pod_status_write "$STATUS_PATH" "ARCHIVED"

pod_runpod_post_success_action "$POST_SUCCESS_ACTION"
pod_status_write "$STATUS_PATH" "SUCCESS"
