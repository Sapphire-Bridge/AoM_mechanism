# MoM Endpoint Plan: Execution Checklist

This checklist follows the locked plan order and enforces the R1 hard fork before extension work.

## Phase 1: Baseline Lock (Run Now)

### Inputs
- `results/paper_lock_20260226T042208Z/clt_raw_comparability_l4_l8_l12.csv`
- `results/paper_lock_20260226T042208Z/clt_raw_comparability_l4_l8_l12.summary.json`
- `data/disamb_pairs.jsonl`

### Command
```bash
python scripts/mom_endpoint_phase1_baseline.py \
  --comparability_csv results/paper_lock_20260226T042208Z/clt_raw_comparability_l4_l8_l12.csv \
  --comparability_summary results/paper_lock_20260226T042208Z/clt_raw_comparability_l4_l8_l12.summary.json \
  --disamb_path data/disamb_pairs.jsonl \
  --bootstrap_n 5000 \
  --seed 42
```

### Required outputs
- `results/mom_endpoint_plan/baseline_metrics.json`
- `results/mom_endpoint_plan/baseline_report.md`

### Phase-1 gate
- Primary metrics present by split (`all_pairs`, `primary_applicable_pairs`, `non_primary_applicable_pairs`):
  - `ΔΔm = Δm_C - Δm_A` + CI + `P(ΔΔm>0)`
  - CRR + CI + `P(CRR>1)` (secondary)
- Hash block present: dataset, tokenizer source path, artifact hashes, git commit.

## Phase 2: Endpoint-Native Telemetry (Code Change)

### Files to edit
- `scripts/clt_raw_comparability.py`
- `scripts/clt_raw_comparability_tasklib.py`

### Add primary endpoint fields
- `P_E_base`, `P_O_base`, `P_rest_base`
- `P_E_patch_A`, `P_O_patch_A`, `P_rest_patch_A`
- `P_E_patch_C`, `P_O_patch_C`, `P_rest_patch_C`
- `dlogPE_A`, `dlogPO_A`, `dlogPE_C`, `dlogPO_C`
- `dPE_A`, `dPO_A`, `dPrest_A`, `dPE_C`, `dPO_C`, `dPrest_C`
- `d_CA_logPE`, `d_CA_logPO`

### Hard asserts in runner
- Applicability assert for primary mechanism rows:
  - all scored candidates single-token
  - expected/other candidate token-ID sets disjoint
  - expected/other candidate counts equal (conservative exclusion for cleaner comparability)
  - same scored position
- Cancellation assert on eligible rows:
  - normalization contribution to margin is numerically ~0
  - default tolerance is `5e-6` (tight enough for float-noise, avoids spurious aborts at `1e-6`)

### Output files
- `results/mom_endpoint_decomp_r1/comparability_endpoint_v2.csv`
- `results/mom_endpoint_decomp_r1/endpoint_pair_aggregates_v2.csv`
- `results/mom_endpoint_decomp_r1/endpoint_decomp_summary_v2.json`
- `results/mom_endpoint_decomp_r1/endpoint_candidate_terms_v2.csv` (reserved for future candidate-level long telemetry)

## Phase 3: R1 Hard Fork Analysis (Required)

### New script
- `scripts/mom_endpoint_decomp_analyze.py`

### Command
```bash
python scripts/mom_endpoint_decomp_analyze.py \
  --comparability_csv results/mom_endpoint_decomp_r1/comparability_endpoint_v2.csv \
  --bootstrap_n 5000 \
  --seed 42
```

### Required outputs
- Pair-bootstrap summaries by layer + split with:
  - `ΔΔm`, CI, `P(ΔΔm>0)` (primary)
  - `d_CA_logPE`, `d_CA_logPO`
  - `dPE`, `dPO`, `dPrest`
  - CRR metrics (secondary)

### Decision fork
- If primary-applicable rows support candidate-set log-odds mechanism:
  - lock main claim to candidate-set log-odds.
- If not:
  - stop and debug decomposition before further claims.

## Phase 4: Precision + Specificity Tests (Required Before Tracing)

### R3 iso-disturbance sweep (delta-form)
- Raw-delta vs SAE-delta only.
- Primary disturbance metric: off-candidate RMS logit change.
- Secondary disturbance metric: full-vocab RMS/JS.

### Matched-basis controls
- Minimum controls:
  - PCA baseline
  - linear AE baseline
- Fairness criterion:
  - match writeback-site reconstruction error (not dimension only).

## Phase 5: Tracing (Conditional)

Run only if all pass:
1. robust `ΔΔm > 0` at R1,
2. advantage survives matched off-candidate disturbance,
3. matched-basis controls do not erase advantage.

Rank and trace by `d_CA`, `dlogPE`, `dlogPO` contributions.

## Phase 6: Multi-token R2 Extension (Optional)

Run only as extension after mainline is settled. Never use to retroactively explain R1.
