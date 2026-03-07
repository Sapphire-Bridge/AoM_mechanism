# Paper verification guide (AoM Prototype repo)

This repo contains **behavioral AoM evaluations** plus **mechanistic probes/interventions**. This guide is meant to let another researcher quickly verify whether a paper’s methods and reported numbers could plausibly have been produced by this codebase and the checked-in `results/` artifacts.

Primary sources in this repo:
- Methods + headline CPT numbers: `aom_paper_technical_facts.md`
- Dataset schemas: `data/README.md`
- Implementation snapshot: `CONTEXT.md`
- Results inventory + quick summaries: `results/results_report.md` (generated 2026-01-11)
- Key CLIs: `aom_eval.py`, `aom_logit_lens.py`, `aom_mechanistic.py`, `aom_sae_check.py`, `aom_sae_sweep.py`

New reviewer-facing reproducibility hooks:
- **Dataset bundle hashing**: `scripts/run_paper.py` writes `DATASET_MANIFEST.json` (SHA-256 for each JSONL). Runners can SHA-gate against it via `--dataset_manifest_path`.
- **DISAMB cue-leakage stratification**: `aom/metrics/cue_leakage.py` + `scripts/audit_disamb_cues.py` + extra DISAMB columns in `aom_eval.csv` (e.g. `disamb_accuracy_overlap_low`, `disamb_accuracy_cue_vulnerable`).
- **CF graded items (optional)**: schema supports `expected_effect="graded"` for partial interventions (behavioral robustness).
- **COH expansion + multi-constraint items (optional)**: generator can produce larger suites (e.g. 80 items) and `n_constraints=2` items.
- **CF/COH causal patching CLIs**: `aom_cf_patching.py` and `aom_coh_patching.py` with sham baselines and relevance controls (shift vs invariant; constraint vs irrelevant).
- **Table regeneration**: `make tables` (or `python scripts/make_tables.py`) generates LaTeX tables from `results/`.

---

## Dataset bundles (paper vs examples)

This repo contains both **checked-in example datasets** and **paper-canonical bundles**. Reviewers should verify which bundle a given result row claims to use.

Two common bundles:

| Bundle | Path | Manifest | Stable bundle ID | DISAMB pairs | CF items | COH main items | COH total rows |
|---|---|---|---|---:|---:|---:|---:|
| Example (checked-in) | `data/` | *(none)* | *(use per-file SHA in run manifests)* | 52 | 60 | 40 | 40 |
| Paper hardened v2 | `data_paper_hardened_v2/` | `data_paper_hardened_v2/DATASET_MANIFEST.json` | `fa2f39387339d26abd45912e31eede3b5f88aac4ed7bff20660262fcb46787ff` | 52 | 140 | 80 | 240 |

Paper-hardened v2 details (seed=0, deterministic generator):
- CF split: 60 `shift`, 60 `invariant`, 20 `graded` (total 140).
- COH groups: 80 `main` + 80 `ablate_relevant` + 80 `ablate_irrelevant` (total 240 JSONL rows).
- COH multi-constraint: 20 `main` items with `n_constraints=2` (and 60 `main` items with `n_constraints=1`).

Expected v2 per-file SHA-256 (from `DATASET_MANIFEST.json`):
- `disamb_pairs.jsonl`: `a9587c185bde2ca83a1fd2d22a62d9d961b2105e7626117240e8aeff207adee1`
- `counterfactual.jsonl`: `e269fbd4fb819591d0bd5932137fcd18037a9ede9e743b7528916ead9f9374bf`
- `coherence.jsonl`: `04e122738582a3d5ea2a7e7f8e38a3c7841c3bab2cc04cf4d24640d5a902fc64`

Notes:
- When `--dataset_manifest_path` is provided, runners record `dataset_bundle_id` in the CSV and run manifest. This ID is stable and content-addressed (depends only on the dataset JSONL hashes).
- The manifest file’s own SHA may change if metadata fields (e.g., timestamps) are rewritten; do not use that as the dataset identifier.

---

## Clean-clone verification (copy/paste runnable)

The following steps are intended to be runnable from a clean clone, with no tribal knowledge.

### 0) Install (pinned environment)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you hit an OpenMP shared-memory error (e.g., `OMP: Error #179: Function Can't open SHM2 failed`), retry with:

```bash
export KMP_USE_SHM=0
```

### 1) Verify the paper dataset bundle (hashes + stable bundle ID)

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
from aom.data.bundle_manifest import compute_bundle_id

d = Path("data_paper_hardened_v2")
m = json.loads((d / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))
print("bundle_id(manifest):", m.get("bundle_id", "<missing>"))
print("bundle_id(recomputed):", compute_bundle_id(m))
for fn in ["disamb_pairs.jsonl", "counterfactual.jsonl", "coherence.jsonl"]:
    h = hashlib.sha256((d / fn).read_bytes()).hexdigest()
    print(fn, h)
PY
```

Expected values for `paper_hardened_v2` are listed above (“Dataset bundles”).

### 2) Produce submission artifacts into a fresh directory

```bash
rm -rf results_submission
python scripts/run_paper.py m1max --results_dir results_submission
```

### 3) Regenerate paper tables (strict mode)

Strict mode fails if any required inputs are missing or missing required provenance columns.

```bash
MAKE_TABLES_STRICT=1 make tables RESULTS_DIR=results_submission TABLES_OUT_DIR=tables_submission
```

### 4) Smoke test (offline)

```bash
python scripts/run_paper.py smoke --results_dir /tmp/aom_smoke_results
make tables RESULTS_DIR=/tmp/aom_smoke_results TABLES_OUT_DIR=/tmp/aom_smoke_tables
```

---

## 1) AoM (“Appearance of Meaning”) evaluation: what is measured

### Suites and dataset sizes
The AoM composite is defined as:
- `aom_composite = mean(disamb_accuracy, cf_shift_direction_accuracy, coh_constraint_accuracy)`.

Bundle-dependent sizes (see “Dataset bundles” above):
- **AoM-DISAMB**: 52 pairs (104 prompt sides). Primary metric: `disamb_accuracy`.
- **AoM-CF (example)**: 60 items (shift-only). Primary metric: `cf_shift_direction_accuracy`.
- **AoM-CF (paper_hardened_v2)**: 140 items total (60 shift / 60 invariant / 20 graded). Primary metric: `cf_shift_direction_accuracy` (over shift items).
- **AoM-COH (example)**: 40 items (main-only). Primary metric: `coh_constraint_accuracy`.
- **AoM-COH (paper_hardened_v2)**: 80 base scenarios (`main`), with paired controls when enabled (240 JSONL rows total).

### Scoring (how numbers are produced)
All three AoM suites are **log-probability scored**, not sampled generation:
- Continuation scoring: mean logprob per token (length-normalized by default).
- Label scoring: `logmeanexp` over that label’s continuation candidates.
- Bootstrap CIs: default `bootstrap_n=1000`, `bootstrap_seed=42`, `ci=0.95` (see `aom_paper_technical_facts.md`).

Paper-verification red flags:
- A paper that reports AoM as sampled generations rather than logprob scoring.
- A paper that reports numbers without identifying the dataset bundle (check `dataset_bundle_id` or per-file SHA-256s).
- A paper that treats `aom_composite` from **DISAMB-only** runs as meaningful (some artifacts are disamb-only; see §2.4).

---

## 2) CPT activation patching (“context-swap”) on AoM-DISAMB

### What the patch is (mechanistic intervention)
Implementation: `aom/metrics/disamb.py` + `aom/interventions/activation_patching.py`.
- Patch site: **decoder block output** (“resid_post” semantics; forward-hook output tensor).
- For each DISAMB pair, patch both directions (A→B and B→A) across layers.
- Patch span: token indices corresponding to the ambiguous substring, found via tokenizer offsets.
- Alignment rule: span lengths must match; by default, token IDs must match; misaligned directions are skipped.

### Effect definition (units)
Let `score(y)` be the label score (logmeanexp over continuation logprobs). Define:
- `margin(scores, y) = score(y) − max_{y'≠y} score(y')`
- For donor→receiver, baseline margin `m_base` is computed on the **receiver** prompt w.r.t. the **donor-expected** label.
- Patched margin `m_patch` is computed under the patch (same expected label).
- Per-layer effect: `effect_ℓ = m_patch − m_base`

Aggregates reported in paper-style runs (see `aom_paper_technical_facts.md`):
- `cpt_mean_max_effect`: mean over directions of `max_ℓ effect_ℓ` (bootstrapped CI)
- `cpt_flip_rate_at_best_layer`: fraction of directions where patch flips receiver from wrong→donor-correct at that direction’s best layer
- `cpt_mean_argmax_layer`: mean of the per-direction argmax layer
- `cpt_mean_sham_max_effect`: same computation with “sham” replacement (receiver→receiver)

### Headline CPT patching numbers (checked-in)
Source: `aom_paper_technical_facts.md` (table), with run artifacts in `results/`.

| Model | Patched/total | Mean max effect | Flip@best | Mean argmax layer | Sham max effect |
|---|---:|---:|---:|---:|---:|
| GPT‑2 | 100/104 | 0.6667 [0.4986, 0.8563] | 0.0500 [0.0100, 0.1000] | 3.96 | 2.28e‑06 [1.46e‑06, 3.15e‑06] |
| Qwen2.5‑0.5B | 100/104 | 1.8612 [1.5227, 2.2634] | 0.1600 [0.0900, 0.2300] | 5.10 | 0.0000 [0.0000, 0.0000] |
| Qwen2.5‑1.5B | 100/104 | 2.4483 [1.9386, 3.0262] | 0.1900 [0.1100, 0.2700] | 6.19 | 7.81e‑05 [0.0000, 2.34e‑04] |
| Qwen2.5‑3B | 100/104 | 3.1137 [2.5340, 3.8002] | 0.2500 [0.1800, 0.3303] | 8.69 | 5.65e‑04 [0.0000, 0.00169] |
| Qwen3‑4B | 100/104 | 2.6861 [2.1532, 3.2953] | 0.1400 [0.0800, 0.2100] | 7.91 | 0.00371 [0.00207, 0.00568] |
| Qwen3‑4B‑Instruct‑2507 | 100/104 | 2.2049 [1.7054, 2.7193] | 0.1300 [0.0700, 0.2000] | 9.23 | 0.00240 [0.00138, 0.00358] |
| Llama‑3.2‑1B | 100/104 | 1.7303 [1.3289, 2.1285] | 0.1800 [0.1100, 0.2600] | 3.28 | 0.00111 [0.00047, 0.00193] |
| Llama‑3.2‑1B‑Instruct | 100/104 | 2.0707 [1.6705, 2.4861] | 0.1900 [0.1200, 0.2602] | 3.49 | 0.00054 [0.00026, 0.00088] |
| Llama‑3.2‑3B | 100/104 | 1.8484 [1.4441, 2.2968] | 0.1300 [0.0700, 0.2000] | 5.26 | 0.00124 [0.00066, 0.00203] |
| Llama‑3.2‑3B‑Instruct | 100/104 | 1.7949 [1.4101, 2.2402] | 0.1300 [0.0700, 0.2000] | 5.83 | 0.00123 [0.00065, 0.00193] |
| Llama‑3.1‑8B‑Instruct | 100/104 | 1.8128 [1.4474, 2.1907] | 0.0800 [0.0300, 0.1400] | 5.99 | 0.00117 [0.00056, 0.00187] |

### Target-specificity control (fixed depth, spatial control)
Implementation: `aom/metrics/disamb.py::compute_cpt_target_specificity_control`.
- Picks `ℓ* = round(0.25 * (L − 1))` unless a fixed layer is provided.
- Compares `E_target` (patch target span) vs `E_ctrl` (patch a same-size off-target span), with deterministic sampling and token-match fallbacks.

Headline numbers (source: `aom_paper_technical_facts.md`):
| Model | ℓ* | E_target | E_ctrl | Δ = E_target−E_ctrl (95% CI) | Win rate |
|---|---:|---:|---:|---:|---:|
| GPT‑2 (124M) | 3 | 0.44 | 0.95 | −0.51 [−0.85, −0.20] | 0.41 |
| Qwen2.5‑0.5B | 6 | 1.25 | 1.37 | −0.13 [−0.65, 0.36] | 0.52 |
| Qwen2.5‑1.5B | 7 | 1.39 | 1.55 | −0.16 [−0.88, 0.42] | 0.52 |
| Qwen2.5‑3B | 9 | 2.15 | 1.27 | +0.87 [0.13, 1.66] | 0.59 |
| Qwen3‑4B | 9 | 1.59 | 1.76 | −0.17 [−1.04, 0.65] | 0.52 |
| Llama‑3.2‑1B | 4 | 1.07 | 1.23 | −0.16 [−0.70, 0.33] | 0.50 |
| Llama‑3.2‑3B | 7 | 1.10 | 1.27 | −0.17 [−0.79, 0.43] | 0.52 |
| Llama‑3.1‑8B | 8 | 0.80 | 1.33 | −0.52 [−1.13, 0.02] | 0.50 |

### DISAMB-only composite pitfall (important)
Files like `results/cpt_specificity_*.csv` run *disamb-only*; CF/COH are zero-filled, so `aom_composite` becomes approximately `disamb_accuracy/3` and **must not** be presented as a full AoM composite.

### CF causal patching (AoM-CF intervention-swap)
CLI: `aom_cf_patching.py`; implementation: `aom/interventions/patching/cf_protocol.py` + `aom/interventions/patching/base.py`.
- Donor/receiver: donor = CF prompt (`x'`), receiver = base prompt (`x`).
- Patch span: token-level diff span between `x` and `x'` (divergent-only by default; can optionally include downstream when total lengths match).
- Primary controls:
  - **Invariant items** (meaning-preserving) should have weaker effects than **shift** items.
  - **Sham patching** (receiver→receiver replacement) should be near zero.
- Key output columns (prefixed by `cf_patch_`): `mean_max_effect`, `mean_sham_max_effect`, stratum splits like `stratum_expected_effect__shift_mean_max_effect`, and effect sizes like `comparison_expected_effect_shift_vs_invariant_cohens_d`.

### COH causal patching (AoM-COH constraint ablation)
CLI: `aom_coh_patching.py`; implementation: `aom/interventions/patching/coh_protocol.py` + `aom/interventions/patching/base.py`.
- Three-condition setup comes from the dataset: `main`, `ablate_relevant`, `ablate_irrelevant`.
- Donor construction: pseudo-ablation by token-ID replacement (keeps main-context positions aligned).
- Primary controls:
  - **Constraint-span patching** should degrade coherence more than **irrelevant-span patching**.
  - **Sham patching** should be near zero.
- Key output columns (prefixed by `coh_patch_`): `mean_max_effect`, `mean_sham_max_effect`, stratum splits like `stratum_condition__constraint_span_mean_max_effect`, and effect sizes like `comparison_condition_constraint_vs_irrelevant_cohens_d`.

---

## 3) Logit lens analysis (expected vs other token)

### What is computed
Implementation: `aom/mechanistic/logit_lens.py`; dataset runner: `scripts/run_logit_lens_dataset.py`.
- Selects a **single-token** continuation for the expected label and a single-token continuation for a comparison label.
- Computes a trace over hidden states `state_index = 0..(n_layers)` at a chosen position (default `-1`).
- At each state, unembeds and records `logit_diff = logit(expected_token) − logit(other_token)`.
- Dataset aggregation: for each `state_index`, computes mean + bootstrap CI and `frac_positive`.

Paper-verification red flags:
- A paper that uses multi-token label continuations but claims it matches this repo’s logit-lens numbers.
- A paper that interprets these as logprobs (they are raw logits).

### Checked-in dataset-level logit-lens numbers
All are over `n_samples = 104` (52 pairs × {a,b}), 95% bootstrap CIs.

| Model | Source file | Peak state (mean, 95% CI) | Final state (mean, 95% CI) |
|---|---|---|---|
| GPT‑2 | `results/logit_lens_disamb_trace.csv` (aggregated from raw trace) | state 10: 6.098 [4.979, 7.327] | state 12: 3.293 [2.528, 4.173] |
| Qwen2.5‑1.5B | `results/logit_lens_disamb_table.csv` | state 25: 8.678 [7.401, 9.918] | state 28: 5.331 [4.561, 6.085] |
| Llama‑3.2‑3B | `results/logit_lens_disamb_table_meta-llama_Llama-3.2-3B_20260111_094245.csv` | state 23: 7.002 [6.169, 7.918] | state 28: 5.150 [4.461, 5.831] |

Single-example trace (one DISAMB prompt):
- `results/logit_lens_trace.csv`: peak at state 10 is `logit_diff ≈ 10.958`, final state 12 is `≈ 7.220`.

---

## 4) Induction-head analysis (synthetic repeats; attention-pattern metric)

### What is computed
Implementation: `aom/mechanistic/induction.py` + `aom/mechanistic/attention_recorder.py`; CLI: `aom_mechanistic.py`.
- Generates synthetic repeated sequences `X|X` with `base_len=64`, `repeats=2` (checked-in results).
- Captures attention weights for selected layers in eager attention mode.
- Induction score: mean attention mass on the induction diagonal (offset=+1) from second-half queries to first-half keys.
- Control baselines:
  - `shuffle`: compares `X|X` vs `perm(X)|X`
  - `offset0`: diagonal offset=0 control
- Reports per head:
  - `induction_score_repeat`, `induction_score_control`, and `induction_advantage = repeat − control`
  - bootstrap CIs over samples (default in artifacts: `bootstrap_n=200` in `aom_mechanistic.py`)

### Checked-in induction results (layer-level mean advantage + best head)
Source files: `results/induction_*.csv` and `results/*_induction.csv` (see `results/results_report.md` for the layer-level summary).

Numbers below use the **layer with max mean(induction_advantage) across heads** and the **single best head** anywhere in the file (same setting: `base_len=64`, `repeats=2`).

| Model | File | Best layer (mean over heads) | Best head (layer, head): advantage (95% CI) |
|---|---|---:|---|
| GPT‑2 (shuffle) | `results/induction_gpt2.csv` | layer 5: 0.1927 | L5H5: 0.9201 [0.9153, 0.9244] |
| GPT‑2 (offset0) | `results/gpt2_induction.csv` | layer 5: 0.1818 | L5H5: 0.9268 [0.9221, 0.9318] |
| Qwen2.5‑0.5B | `results/induction_Qwen_Qwen2_5-0_5B.csv` | layer 16: 0.4904 | L16H3: 0.9696 [0.9664, 0.9728] |
| Qwen2.5‑3B | `results/induction_Qwen_Qwen2_5-3B.csv` | layer 32: 0.3347 | L20H1: 0.9809 [0.9761, 0.9853] |
| Qwen3‑4B | `results/induction_Qwen_Qwen3-4B.csv` | layer 29: 0.1178 | L13H13: 0.9623 [0.9577, 0.9669] |
| Llama‑3.2‑3B | `results/induction_meta-llama_Llama-3_2-3B.csv` | layer 14: 0.0964 | L14H22: 0.9339 [0.9294, 0.9384] |

Paper-verification red flags:
- A paper that claims these induction numbers without using eager attention weights (or without a comparable attention-capture method).
- A paper that reports “induction score” but uses a different diagonal definition than offset +1 over the `X|X` second-half→first-half slice.

---

## 5) SAE analysis (Gemma Scope SAEs): sterility, feature-space CPT, and threshold sweeps

This repo contains **SAE-backed interventions** (Gemma Scope SAEs) in `aom/interventions/`.

### 5.1 Sterility (roundtrip) check
CLI: `aom_sae_check.py`; implementation: `aom/interventions/sae_sterility.py`.
- Roundtrip mode reconstructs the block output via SAE and replaces it in the forward pass.
- Reports:
  - DISAMB accuracy delta (baseline vs roundtrip)
  - mean per-token KL(p_base || p_roundtrip)
  - recon MSE on a captured block output tensor
- Gate: `abs(delta_acc) <= tolerance` and `mean_token_kl <= kl_tolerance`.

Note: sterility JSON outputs are *not* checked in under `results/` here; the checked-in SAE artifacts below are from the sweep and SAE-CPT runs.

### 5.2 SAE feature-space CPT (context-swap analogue)
Implementation: `aom/metrics/sae_patching.py` (replaces SAE features at the target span with donor features).

Checked-in Gemma‑2‑2B comparisons:
- Hidden-state CPT sweep (selected layers): `results/gemma2b_cpt_hidden.csv`
  - `cpt_mean_max_effect = 0.8978` [0.7199, 1.0813]
  - `cpt_flip_rate_at_best_layer = 0.0865` [0.0385, 0.1346]
  - Per-layer effects include:
    - `cpt_effect_layer_15 = 0.2517`
    - `cpt_effect_layer_18 = 0.0939`
- SAE CPT at layer 15 only: `results/gemma2b_sae_cpt_layer15.csv`
  - `sae_cpt_mean_max_effect = 0.2566` [0.0600, 0.4584]
  - `sae_cpt_flip_rate_at_best_layer = 0.0288` [0.0000, 0.0673]
  - `sae_cpt_mean_sham_max_effect ≈ 1.21e-06`
- SAE CPT at layer 18 only: `results/gemma2b_sae_cpt_layer18.csv`
  - `sae_cpt_mean_max_effect = 0.0908` [−0.0769, 0.2384]
  - `sae_cpt_flip_rate_at_best_layer = 0.0288` [0.0000, 0.0577]
  - `sae_cpt_mean_sham_max_effect ≈ 2.60e-07`

### 5.3 SAE threshold sweep (feature pruning + controls)
CLI: `aom_sae_sweep.py`; implementation: `aom/interventions/sae_sweep.py`.
- Policy: `RelativeThresholdPolicy` (keep high-salience features; `scale_mode=quantile`, `quantile=0.95` in checked-in runs).
- Controls per threshold:
  - `random_matched_active`: random mask with matched sparsity
  - `anti_keep_low`: keep low features instead (anti-control; flags `h2_falsified` when anti is ≥ primary on |Δexpected_nll|)
  - `off_target_token`: apply same policy at an off-target span

Checked-in sweeps (Gemma‑2‑2B):
- Layer 15: `results/sae_sweep_layer15.csv` (+ manifest `results/sae_sweep_layer15.json`)
  - baseline: accuracy 0.8846, mean_expected_nll 7.4345 (52 pairs)
  - example operating point (threshold=1.0): `threshold` accuracy 0.8173 (Δacc −0.0673), Δexpected_nll +0.6226, sparsity_gain 0.7747
- Layer 18: `results/sae_sweep_layer18.csv` (+ manifest `results/sae_sweep_layer18.json`)
  - baseline: accuracy 0.8846, mean_expected_nll 7.4345 (52 pairs)
  - example operating point (threshold=1.0): `threshold` accuracy 0.8462 (Δacc −0.0385), Δexpected_nll +0.4238, sparsity_gain 0.6831

Paper-verification red flags:
- A paper claiming SAE results without specifying the SAE source (“Gemma Scope”) and layer/run configuration.
- A paper interpreting `mean_expected_nll` as token-level cross-entropy; in this repo it is `-score(expected_label)` where score is the DISAMB label log-score.

---

## 6) Concrete “is the paper based on this repo?” checklist

Use this as a literal audit checklist:

1) **Datasets & sizes**
   - DISAMB pairs = 52; CF/COH sizes are bundle-dependent (verify via `DATASET_MANIFEST.json` + `bundle_id` / `dataset_bundle_id`).
   - Paper bundle `paper_hardened_v2` (seed=0): CF=140 (60 shift / 60 invariant / 20 graded), COH main=80 with controls → 240 JSONL rows.
   - Example bundle `data/`: CF=60 (shift-only), COH=40 (main-only).
   - Schema matches `data/README.md` (e.g., DISAMB has `target` and `target_occurrence`).

2) **AoM scoring**
   - Uses deterministic logprob scoring, not sampled text.
   - Uses length-normalized continuation logprobs by default.
   - Uses logmeanexp aggregation across a label’s continuation set.
   - Uses bootstrap over the correct unit (pairs/items/base_id), typically `bootstrap_n=1000`.

3) **CPT patching**
   - Patch site is block output (“resid_post”), not attention weights.
   - Effect is margin shift `m_patch − m_base` on the donor-expected label.
   - Reports sham baselines near zero.
   - If the paper reports the table in §2.3, it should match `aom_paper_technical_facts.md` and the corresponding `results/*.csv`.

4) **Target specificity**
   - Uses fixed depth `ℓ* = round(0.25*(L−1))` and local spatial controls as described in §2.3.
   - If a paper reports the Δ table, it should match `aom_paper_technical_facts.md` and `results/cpt_specificity_*.csv`.

5) **Logit lens**
   - Uses single-token continuations (or explicitly explains deviations).
   - If referencing dataset-level curves, the peak/final numbers should match §3.3 and the corresponding `results/logit_lens_disamb_table*.csv` or `results/logit_lens_disamb_trace.csv`.

6) **Induction heads**
   - Uses repeated-sequence attention-pattern metric (offset +1 diagonal over `X|X` second→first slice).
   - Reports baseline/control mode (`shuffle` or `offset0`), base_len=64, repeats=2 (or states differences).
   - If referencing “best layer mean advantage”, it should match `results/results_report.md` induction summary and the numbers in §4.2.

7) **SAE analysis**
   - Names the SAE bundle (Gemma Scope) and the SAE run (e.g. `average_l0_308`) + layer.
   - Uses sterility checks or reports equivalent safeguards (delta accuracy + KL + recon MSE).
   - If reporting threshold sweeps, includes controls (`random_matched_active`, `anti_keep_low`, `off_target_token`) and does not over-interpret `mean_expected_nll`.
