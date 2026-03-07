#!/usr/bin/env bash
set -euo pipefail

# Memory-friendly MPS sweep (sequential model loading) that includes CPT patching + specificity.
# Default writes results to: results/mps_sweep_<timestamp>.csv
#
# Usage:
#   DATA_DIR=data_paper_20260104_005224 bash scripts/run_mps_sweep_run2.sh
#   LOCAL_FILES_ONLY=1 DATA_DIR=data bash scripts/run_mps_sweep_run2.sh

DATA_DIR="${DATA_DIR:-data}"
TS="${TS:-$(date +"%Y%m%d_%H%M%S")}"
OUT_CSV="${OUT_CSV:-results/mps_sweep_${TS}.csv}"
LOG_DIR="${LOG_DIR:-results/logs/mps_sweep_${TS}}"

DISAMB_PATH="${DISAMB_PATH:-${DATA_DIR}/disamb_pairs.jsonl}"
CF_PATH="${CF_PATH:-${DATA_DIR}/counterfactual.jsonl}"
COH_PATH="${COH_PATH:-${DATA_DIR}/coherence.jsonl}"

if [[ ! -f "${DISAMB_PATH}" ]]; then
  echo "Missing DISAMB_PATH: ${DISAMB_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CF_PATH}" ]]; then
  echo "Missing CF_PATH: ${CF_PATH}" >&2
  exit 1
fi
if [[ ! -f "${COH_PATH}" ]]; then
  echo "Missing COH_PATH: ${COH_PATH}" >&2
  exit 1
fi

echo "[config] DATA_DIR=${DATA_DIR}"
echo "[config] DISAMB_PATH=${DISAMB_PATH}"
echo "[config] CF_PATH=${CF_PATH}"
echo "[config] COH_PATH=${COH_PATH}"
echo "[config] OUT_CSV=${OUT_CSV}"
echo "[config] LOG_DIR=${LOG_DIR}"

cmd=(python scripts/run_mps_sweep.py \
  --fresh \
  --models \
    gpt2 \
    Qwen/Qwen2.5-0.5B \
    Qwen/Qwen2.5-1.5B \
    Qwen/Qwen2.5-3B \
    Qwen/Qwen3-4B \
    Qwen/Qwen3-4B-Instruct-2507 \
    meta-llama/Llama-3.2-1B \
    meta-llama/Llama-3.2-1B-Instruct \
    meta-llama/Llama-3.2-3B \
    meta-llama/Llama-3.2-3B-Instruct \
    meta-llama/Meta-Llama-3.1-8B \
    meta-llama/Meta-Llama-3.1-8B-Instruct \
  --seeds 0 1 2 \
  --device mps \
  --torch_dtype float16 \
  --attn_implementation eager \
  --logprobs_dtype float32 \
  --bootstrap_n 200 \
  --patch_layers auto \
  --disamb_path "${DISAMB_PATH}" \
  --cf_path "${CF_PATH}" \
  --coh_path "${COH_PATH}" \
  --out_csv "${OUT_CSV}" \
  --log_dir "${LOG_DIR}")

if [[ "${LOCAL_FILES_ONLY:-}" == "1" ]]; then
  cmd+=(--local_files_only)
fi

echo "[run] ${cmd[*]}"
exec "${cmd[@]}"
