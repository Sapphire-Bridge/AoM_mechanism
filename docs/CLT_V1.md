# CLT v1 Implementation Spec (Inference-Only, Publication-Oriented)

## Goal
Ship a non-breaking Cross-Layer Transcoder (CLT) patching path that can support a minimal, falsifiable mechanistic claim in this repository.

## Minimal Claim (v1)
On DISAMB, CLT feature replacement recovers a non-trivial fraction of raw activation patching effect, with strong control separation and identity invariance.

## Primary Metrics
- `raw_cpt_effect`: raw CPT effect at the exact CLT `writeback_site` and same `(layer, span)` intervention target.
- `clt_cpt_effect`: CLT-mediated CPT effect.
- `clt_sham_effect`: CLT sham/no-op effect.
- `crr` (CLT Recovery Ratio): `clt_cpt_effect / max(eps, raw_cpt_effect)`.

## Scope
- Inference-time CLT patching only.
- No CLT training pipeline in this repo for v1.
- DISAMB first; broader tasks can be added after controls pass.

## Non-Goals (v1)
- No claim of complete circuit coverage.
- No cross-task universality claim.
- No dependency on disk activation cache.

## Contract: Hookpoint and Semantics
Every CLT run must log these fields in outputs and manifest:
- `clt_encode_site`: where encoder reads (for example `resid_pre` at layer `l`).
- `clt_decode_site`: where decoder reconstructs target tensor (for example `mlp_out` at layer `l`).
- `clt_writeback_site`: where patched tensor is injected.
- `clt_error_policy`: must be `preserve_error` in v1.
- `clt_layers`: explicit layer list (no implicit all-layers mode).
- `clt_site_mode`: `same_site_v1` for this release.

For v1, enforce:
- `encode_site == decode_site == writeback_site`.
- Cross-site patching is deferred to v1.1 with an explicit `PatchContext` mechanism.

## Critical Mechanism: Error-Preserving Writeback (Must-Have)
Let:
- `x`: tensor at `encode_site`.
- `y`: tensor at `decode_site` (same as `writeback_site` in v1).

For each patched site:
- Compute baseline latent `z = E(x)`.
- Compute baseline reconstruction `y_hat = D(z)`.
- Compute residual error `e = y - y_hat`.
- Build edited reconstruction `y_hat_prime = D(z_prime)` where `z_prime` is the modified latent.
- Write back `y_prime = y_hat_prime + e`.

Rationale:
- Identity edits should preserve model behavior.
- Intervention effect is attributed to latent edits, not reconstruction gap.

## Controls (All Required)
- Identity control: execute full patch code path with `z_prime = z_receiver`; output should match baseline.
- Permutation donor control: donor examples are permuted.
- Span-miss control: patch same layer but irrelevant span.
- Sham control: run donor/receiver selection and scaling pipeline, but override to `z_prime = z_receiver` at writeback.

## Acceptance Gates (v1)
- Gate 1: Identity invariance.
  - Logit drift from identity patch is near zero (predefined tolerance by dtype).
- Gate 2: Reconstruction telemetry sane at tested layers.
  - Report `recon_rel_l2` and `recon_cos` distributions.
- Gate 3: Control separation.
  - Target effect materially exceeds identity/permutation/span-miss/sham effects.
- Gate 4: Statistical robustness.
  - Bootstrap CI for target effect excludes zero on held-out evaluation split.
- Gate 5: Recovery ratio reported.
  - `crr` and CI written for each run using joint bootstrap over `(raw_cpt_effect, clt_cpt_effect)`.
- Gate 6: Reproducibility.
  - Model and CLT artifacts pinned and hashed in manifest.

## Data-Split Policy (Avoid Layer Selection Bias)
- Use a calibration split to choose layers and scale.
- Use a disjoint held-out split to report effects.
- If best-layer reporting is used, it must be selected on calibration split only.
- Manifest must include `clt_selected_layers_calib` and `clt_best_layer_calib`.

## Calibration Policy
Scale selection is deterministic and logged:
- Either feature-norm matching or non-destructive drift threshold strategy.
- Required outputs:
  - selected scale,
  - calibration IDs,
  - thresholds and criterion,
  - dtype policy.

## File/Module Plan
- `aom/interventions/clt_adapter.py`
  - CLT protocol + config dataclasses.
- `aom/interventions/clt_loader.py`
  - local/HF load path, metadata, artifact pinning support.
  - supports `activation=jumprelu` with optional per-feature `threshold`.
  - `pre_encoder_bias` is opt-in and defaults to `False` for backward compatibility.
- `aom/interventions/clt_patch.py`
  - primitive patching ops + error-preserving writeback.
- `aom/metrics/clt_cpt.py`
  - DISAMB CLT CPT metric pipeline.
- `aom_clt_check.py`
  - preflight checker: identity test + recon telemetry + calibration.
- `aom_eval.py`
  - optional `--run_clt_patching` stage and `clt_*` outputs.

## Naming/Output Schema
- Prefix all new row fields with `clt_`.
- Keep existing `cpt_*` and `sae_cpt_*` unchanged.
- Effect fields used for CRR are nonnegative magnitudes (sign conventions documented in row metadata).
- Required CLT summary fields:
  - `clt_cpt_mean_max_effect`
  - `clt_cpt_mean_norm_max_effect`
  - `clt_cpt_mean_sham_max_effect`
  - `clt_cpt_flip_rate_at_best_layer`
  - `clt_cpt_mean_argmax_layer`
  - `clt_cpt_n_directions_total`
  - `clt_cpt_n_directions_patched`
  - `clt_cpt_n_directions_skipped_misaligned`
  - `clt_recovery_ratio`
  - `clt_recovery_ratio_ci_low`
  - `clt_recovery_ratio_ci_high`
  - `clt_selected_layers_calib`
  - `clt_best_layer_calib`

## Manifest Requirements
Include at minimum:
- `model_id`, `model_revision`, `model_commit_hash` (if available),
- `clt_repo_or_path`, `clt_revision`,
- sha256 for CLT weight files used,
- dataset manifest path and hash,
- encode/decode/writeback sites,
- error policy,
- calibration settings and chosen scale.

## Performance Constraints
- Cache donor encodes per `(example_id, layer, span)` in-memory during run.
- Keep smoke mode bounded (`--n_examples`, fixed layers, low bootstrap count).
- Do not require persistent activation caches in v1.

## Test Plan
- Unit tests:
  - loader path resolution and metadata integrity,
  - shape checks and dtype behavior,
  - error-preserving identity invariance.
- Metric tests:
  - CLT CPT runs on tiny DISAMB sample and writes required columns.
  - control effects stay near zero in synthetic sanity setup.
- Integration tests:
  - `aom_eval.py --run_clt_patching` executes offline with local artifacts.

## Recommended Build Order
1. `clt_adapter.py` + `clt_loader.py`.
2. `clt_patch.py` with error-preserving writeback and identity test.
3. `aom_clt_check.py` with recon telemetry + calibration.
4. `clt_cpt.py` metrics and bootstrap summaries.
5. `aom_eval.py` integration and manifest/reporting hooks.
6. smoke/integration tests.

## Success Criteria
CLT v1 is considered ready when:
- all acceptance gates pass on DISAMB held-out split,
- output schema is stable and reproducible,
- `clt_*` metrics can be compared directly against `cpt_*` and `sae_cpt_*` in one run row.
