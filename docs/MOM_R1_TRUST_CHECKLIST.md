# Mechanistic R1 Trust Checklist

Use this **before trusting a major run** or merging changes that affect the mechanism claim.

## A. Hard gates - must all be true

- [ ] **1. One shared tolerance constant**
  - Check: runner, tasklib runner, and analyzer all import the same residual-tolerance constant from `aom.metrics.primary_logodds`.
  - Pass: no stray hard-coded `1e-6` / `5e-6` defaults remain.

- [ ] **2. One shared primary-applicability definition**
  - Check: baseline script, DISAMB runner, and tasklib runner all call `evaluate_primary_applicability(...)`.
  - Pass: no duplicated eligibility logic exists elsewhere.

- [ ] **3. `same_scored_position` is computed, not assumed**
  - Check: `evaluate_primary_applicability(...)` derives it from `scored_positions`.
  - Pass: no `same_scored_position = True` placeholders remain.

- [ ] **4. New-schema enforcement is active**
  - Check: analyzer fails on old CSVs or missing required columns.
  - Pass: `_load_rows(...)` raises on schema mismatch.

- [ ] **5. Exact primary log-odds identity holds on primary-applicable rows**
  - Check: `primary_base_logodds_residual`, `primary_delta_m_residual_A`, `primary_delta_m_residual_C`, `primary_cancellation_pass`.
  - Pass: all primary-applicable rows pass tolerance; no silent violations.

- [ ] **6. Primary metrics are only reported for fully primary-complete pairs**
  - Check: mixed pairs produce `NaN` for pair-level primary mechanism metrics.
  - Pass: no partial-pair averaging leaks into primary summaries.

## B. Strong confidence checks - should be true before using results in the paper

- [ ] **7. Diagnostic logZ is named as diagnostic everywhere**
  - Check: fields and summaries use `d_CA_diag_logz` / `supports_primary_diag_logz_nonzero`.
  - Pass: no variable, table, or summary implies endpoint-matched z attribution unless actually implemented.

- [ ] **8. Tasklib sign semantics are locked**
  - Check: `signed_delta_margin_from_logodds(...)` is the only sign helper used; tests cover `effect_sign = +1` and `-1`.
  - Pass: tasklib signed deltas match expected analytical values.

- [ ] **9. Provenance is fully recorded**
  - Check artifacts include:
    - dataset hash
    - summary/artifact hashes
    - tokenizer resolved path
    - tokenizer file hashes
    - git commit
    - bootstrap seed / n / tolerance
  - Pass: all are present in baseline and analysis summaries.

- [ ] **10. End-to-end smoke passes on the new schema**
  - Check:
    - runner emits new primary fields
    - analyzer consumes new-schema smoke output
    - baseline script reruns cleanly
  - Pass: no import-path, schema, or naming-contract break remains.

## Claim-readiness gates

## Do **not** trust the mechanism result unless:

- [ ] 1-6 are all checked

## Do **not** claim candidate-set mechanism in the manuscript unless:

- [ ] primary-applicable rows show tiny residuals
- [ ] analyzer reports `supports_candidate_set_logodds_accounting = True`
- [ ] `Deltadeltam` is evaluated on `primary_applicable_pairs`, not a looser split

## Do **not** claim anything about logZ beyond "diagnostic" unless:

- [ ] endpoint-matched z attribution is actually implemented
- [ ] or a separate extension regime explicitly supports it

## Do **not** claim "precision superiority" unless:

- [ ] iso-disturbance R3 is complete
- [ ] advantage survives matched off-candidate disturbance

## Do **not** claim "SAE-specificity" unless:

- [ ] matched PCA / linear-AE controls fail to match SAE

## Do **not** claim feature/head mechanism unless:

- [ ] tracing is stable across splits
- [ ] ablations move `d_CA`, not just absolute effect

## Minimal run-acceptance note

A run is **usable for interpretation** only if all of these are true:

- [ ] `n_rows_primary_cancellation_fail == 0`
- [ ] `n_rows_primary_residual_violations == 0`
- [ ] primary-applicable split is present
- [ ] analyzer summary generated from the new-schema CSV
- [ ] provenance block is complete
