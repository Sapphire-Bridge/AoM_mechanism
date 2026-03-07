#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CF_PATH="data_paper_hardened_v2/counterfactual.jsonl"
SPAN_MODE="left_aligned_truncated"
EFFECTS="shift,invariant"

echo "=== GPT-2 ==="
python aom_cf_patching.py \
  --config configs/cf_patching_gpt2_paper.yaml \
  --cf_path "$CF_PATH" \
  --span_mode "$SPAN_MODE" \
  --include_expected_effects "$EFFECTS" \
  --sweep_seeds 0 \
  --run_name paper_cf_patching_gpt2_trunc_si

echo "=== Qwen2.5-0.5B ==="
python aom_cf_patching.py \
  --config configs/cf_patching_qwen25_05b_paper.yaml \
  --cf_path "$CF_PATH" \
  --span_mode "$SPAN_MODE" \
  --include_expected_effects "$EFFECTS" \
  --sweep_seeds 0 \
  --run_name paper_cf_patching_qwen25_05b_trunc_si

echo "=== Qwen2.5-1.5B ==="
python aom_cf_patching.py \
  --config configs/cf_patching_qwen25_15b_paper.yaml \
  --cf_path "$CF_PATH" \
  --span_mode "$SPAN_MODE" \
  --include_expected_effects "$EFFECTS" \
  --sweep_seeds 0 \
  --run_name paper_cf_patching_qwen25_15b_trunc_si

echo "=== Qwen2.5-3B ==="
python aom_cf_patching.py \
  --config configs/cf_patching_qwen25_3b_paper.yaml \
  --cf_path "$CF_PATH" \
  --span_mode "$SPAN_MODE" \
  --include_expected_effects "$EFFECTS" \
  --sweep_seeds 0 \
  --run_name paper_cf_patching_qwen25_3b_trunc_si

echo "=== All done ==="
