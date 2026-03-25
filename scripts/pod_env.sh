#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"

die() {
  echo "[error] $*" >&2
  exit 1
}

pod_bool_enabled() {
  [[ "${1:-0}" == "1" ]]
}

pod_status_write() {
  local status_path="$1"
  local value="$2"
  printf '%s\n' "$value" > "$status_path"
}

pod_require_empty_dir() {
  local path="$1"
  if [[ -e "$path" && ! -d "$path" ]]; then
    die "Path exists but is not a directory: $path"
  fi
  if [[ -d "$path" ]] && find "$path" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    die "Directory must start empty: $path"
  fi
}

pod_sha256_cmd() {
  if command -v sha256sum >/dev/null 2>&1; then
    echo "sha256sum"
    return 0
  fi
  if command -v shasum >/dev/null 2>&1; then
    echo "shasum -a 256"
    return 0
  fi
  die "Missing sha256 tool (expected sha256sum or shasum)"
}

pod_archive_path() {
  local result_path="$1"
  printf '%s.tar.gz' "$result_path"
}

pod_checksum_path() {
  local archive_path="$1"
  printf '%s.sha256' "$archive_path"
}

pod_archive_result() {
  local result_path="$1"
  local archive_path="$2"
  local checksum_path="$3"
  local parent_dir
  local base_name
  parent_dir="$(dirname "$result_path")"
  base_name="$(basename "$result_path")"
  [[ -e "$result_path" ]] || die "Missing result path for archive: $result_path"
  mkdir -p "$(dirname "$archive_path")"
  tar -czf "$archive_path" -C "$parent_dir" "$base_name"
  local hash_cmd
  hash_cmd="$(pod_sha256_cmd)"
  # shellcheck disable=SC2086
  $hash_cmd "$archive_path" > "$checksum_path"
}

pod_runpod_post_success_action() {
  local action="$1"
  case "$action" in
    none)
      return 0
      ;;
    auto)
      if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
        runpodctl stop pod "$RUNPOD_POD_ID"
      fi
      return 0
      ;;
    stop)
      [[ -n "${RUNPOD_POD_ID:-}" ]] || die "RUNPOD_POD_ID is required for post-success action 'stop'"
      command -v runpodctl >/dev/null 2>&1 || die "runpodctl is required for post-success action 'stop'"
      runpodctl stop pod "$RUNPOD_POD_ID"
      return 0
      ;;
    terminate)
      [[ -n "${RUNPOD_POD_ID:-}" ]] || die "RUNPOD_POD_ID is required for post-success action 'terminate'"
      command -v runpodctl >/dev/null 2>&1 || die "runpodctl is required for post-success action 'terminate'"
      runpodctl remove pod "$RUNPOD_POD_ID"
      return 0
      ;;
    *)
      die "Unknown post-success action: $action"
      ;;
  esac
}

pod_launch_background() {
  local log_path="$1"
  local pid_path="$2"
  shift 2

  nohup "$@" > "$log_path" 2>&1 < /dev/null &
  local pid="$!"
  echo "$pid" > "$pid_path"
  echo "log: $log_path"
  echo "pid_file: $pid_path"
  echo "pid: $pid"
  echo "monitor: tail -f \"$log_path\""
}

[[ -d "$REPO_ROOT" ]] || die "Repo root does not exist: $REPO_ROOT"
[[ -f "$REPO_ROOT/.venv/bin/activate" ]] || die "Missing virtualenv activate script: $REPO_ROOT/.venv/bin/activate"
[[ -f "$REPO_ROOT/scripts/run_one_result_check.py" ]] || die "Repo root is missing scripts/run_one_result_check.py: $REPO_ROOT"
[[ -f "$REPO_ROOT/scripts/run_paper.py" ]] || die "Repo root is missing scripts/run_paper.py: $REPO_ROOT"
[[ -f "$REPO_ROOT/scripts/run_mom_paper.py" ]] || die "Repo root is missing scripts/run_mom_paper.py: $REPO_ROOT"

cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"

export REPO_ROOT
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
# Backward compatibility for older transformers cache lookup paths.
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" "$REPO_ROOT/results"
