#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

DATA="data_paper_hardened_v2"
OUTDIR="results/overnight_$(date +%Y%m%d)"
mkdir -p "$OUTDIR"

# Llama family only, smallest to largest
MODELS=(
  meta-llama/Llama-3.2-1B
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B
  meta-llama/Llama-3.2-3B-Instruct
  meta-llama/Meta-Llama-3.1-8B
)

# Patching: 3B base + 8B
PATCH_MODELS=(
  meta-llama/Llama-3.2-3B
  meta-llama/Meta-Llama-3.1-8B
)

echo "=== Starting overnight run: $(date) ==="
echo "=== Results dir: $OUTDIR ==="

# --- Phase 1: Behavioral eval, one model at a time ---
for m in "${MODELS[@]}"; do
  slug="${m//\//_}"
  csv="$OUTDIR/behavioral_${slug}.csv"
  echo ""
  echo "=== Behavioral: $m ($(date)) ==="
  python aom_eval.py \
    --models "$m" \
    --device auto \
    --attn_implementation sdpa \
    --disamb_path "$DATA/disamb_pairs.jsonl" \
    --cf_path "$DATA/counterfactual.jsonl" \
    --coh_path "$DATA/coherence.jsonl" \
    --dataset_manifest_path "$DATA/DATASET_MANIFEST.json" \
    --bootstrap_n 1000 \
    --bootstrap_seed 42 \
    --ci 0.95 \
    --csv_path "$csv" \
    2>&1 | tee "$OUTDIR/behavioral_${slug}.log"
  echo "=== Behavioral done: $m -> $csv ==="
done

# --- Phase 2: DISAMB layer-sweep patching ---
for m in "${PATCH_MODELS[@]}"; do
  slug="${m//\//_}"
  echo ""
  echo "=== DISAMB patching: $m ($(date)) ==="
  python aom_eval.py \
    --models "$m" \
    --device auto \
    --attn_implementation eager \
    --disamb_path "$DATA/disamb_pairs.jsonl" \
    --cf_path "$DATA/counterfactual.jsonl" \
    --coh_path "$DATA/coherence.jsonl" \
    --dataset_manifest_path "$DATA/DATASET_MANIFEST.json" \
    --bootstrap_n 1000 \
    --bootstrap_seed 42 \
    --ci 0.95 \
    --run_patching \
    --csv_path "$OUTDIR/patching_${slug}.csv" \
    2>&1 | tee "$OUTDIR/patching_${slug}.log"
  echo "=== DISAMB patching done: $m ==="
done

# --- Phase 3: CF patching (left_aligned_truncated) ---
for m in "${PATCH_MODELS[@]}"; do
  slug="${m//\//_}"
  echo ""
  echo "=== CF patching: $m ($(date)) ==="
  python aom_cf_patching.py \
    --models "$m" \
    --device auto \
    --attn_implementation eager \
    --cf_path "$DATA/counterfactual.jsonl" \
    --span_mode left_aligned_truncated \
    --include_expected_effects shift,invariant \
    --sweep_seeds 0 \
    --bootstrap_n 1000 \
    --bootstrap_seed 42 \
    --ci 0.95 \
    --run_name "cf_patching_trunc_${slug}" \
    --results_dir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/cf_patching_${slug}.log"
  echo "=== CF patching done: $m ==="
done

echo ""
echo "=== All done: $(date) ==="
echo "Results in: $OUTDIR/"
ls -la "$OUTDIR"/*.csv 2>/dev/null || echo "No CSVs found"
