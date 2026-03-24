# CONTEXT-005

## Scope

Files covered here:

- `aom_feature_families.py`
- `aom_clt_topk_recovery.py`
- `aom_clt_feature_analysis.py`
- `aom_clt_head_attribution.py`
- `aom_logit_lens.py`
- `mom_circuits_tlens.py`
- `aom/mechanistic/backends/transformer_lens.py`
- `scripts/export_mom_flagship_manifest.py`
- `scripts/run_mom_flagship.py`
- `scripts/run_scaling_study.py`
- `configs/scaling_study.yaml`
- `scripts/run_logit_lens_dataset.py`

## Architecture Overview

These tools extend the core paper path into richer mechanistic analysis and model-family comparison.

- CLT feature tooling extracts, ranks, and groups learned latent structure.
- Head-attribution tooling bridges model internals into attention-head level analyses.
- TransformerLens-backed scripts provide a second mechanistic surface for flagship circuit work.
- Scaling-study tooling turns the same evaluation ideas into multi-model sweeps driven by config.

This layer depends heavily on the reproducibility and patching primitives from `CONTEXT-001` and `CONTEXT-002`, but it is not always required for baseline reproduction.

## Key Flows

### 1. CLT recovery and feature analysis

- `aom_clt_topk_recovery.py` uses CLT-based intervention outputs to compute top-k recovery style summaries.
- `aom_clt_feature_analysis.py` performs broader feature-level analysis over CLT representations.
- `aom_feature_families.py` groups or scores feature families for higher-level interpretation.

These scripts are useful when a result has moved beyond "is there an effect?" to "which learned latent directions seem to carry it?"

### 2. Head attribution and TransformerLens bridge

- `aom_clt_head_attribution.py` scores or ablates head contributions using a TransformerLens-style mechanistic view.
- `aom/mechanistic/backends/transformer_lens.py` is the adapter layer that lets the repo’s abstractions talk to TransformerLens-oriented logic.
- `mom_circuits_tlens.py` consumes flagship manifests and can operate either from manifest-selected heads or ranked selections.

The important implication is that layer/head indexing and tokenization assumptions must remain consistent across the Hugging Face and TransformerLens paths.

### 3. Flagship manifest export and execution

- `scripts/export_mom_flagship_manifest.py` exports a DISAMB-derived flagship manifest including selected layers, heads, and dataset-manifest context.
- `scripts/run_mom_flagship.py` runs canonical flagship evaluations while checking manifest protocol hashes.
- This creates a traceable handoff from earlier discovery analyses into curated flagship experiments.

### 4. Scaling study path

- `scripts/run_scaling_study.py` reads `configs/scaling_study.yaml` and executes a per-model study battery.
- It writes per-model and study-level manifests rather than leaving the study state implicit in shell history.

### 5. Logit-lens style analysis

- `aom_logit_lens.py` and `scripts/run_logit_lens_dataset.py` provide a separate interpretability surface for dataset-wide probing.
- These tools are supportive rather than central to the main paper runner, but they follow the same expectation that outputs remain attributable.

## Research Invariants

- Extended analysis tools should not invent their own provenance conventions. They should continue writing manifests or equivalent sidecar metadata.
- Exported flagship manifests are a contract. Downstream TL or head-attribution runs should consume them rather than silently re-deriving selections.
- Layer and head numbering must stay aligned across backends. Index mismatches can yield compelling but invalid circuit stories.
- Scaling-study comparisons are only meaningful when tokenizer/model revisions and dataset manifests remain controlled across models.

## Interfaces

Main scripts and tools:

- `aom_clt_topk_recovery.py`
- `aom_clt_feature_analysis.py`
- `aom_feature_families.py`
- `aom_clt_head_attribution.py`
- `mom_circuits_tlens.py`
- `scripts/export_mom_flagship_manifest.py`
- `scripts/run_mom_flagship.py`
- `scripts/run_scaling_study.py`
- `aom_logit_lens.py`
- `scripts/run_logit_lens_dataset.py`

Key shared dependency:

- `aom/mechanistic/backends/transformer_lens.py`

## Cached Artifacts And Provenance

- These paths often depend on precomputed CLT bundles, flagship manifests, and model caches.
- Because many of the analyses are expensive, manifested intermediate outputs are especially important for reproducibility.
- Study-level and flagship-level manifests are the correct starting point for auditing these tools.

## Operational Notes / Gotchas

- TransformerLens-backed code adds another backend boundary. Treat backend adapters as correctness-sensitive, not as convenience wrappers.
- Head and layer selections are easy to desynchronize from the source dataset or protocol version. Prefer exported manifests over manual CLI reconstruction.
- Scaling-study scripts increase the chance of subtle per-model drift. Audit the config and the emitted manifests before trusting aggregate comparisons.
- Some of these tools are support analyses rather than paper-critical mainline outputs. Keep that distinction clear during triage.

## Claim Traceability

- CLT feature and recovery claims: `aom_clt_topk_recovery.py`, `aom_clt_feature_analysis.py`, `aom_feature_families.py`
- Head attribution and TL circuit claims: `aom_clt_head_attribution.py`, `aom/mechanistic/backends/transformer_lens.py`, `mom_circuits_tlens.py`
- Flagship selection and execution claims: `scripts/export_mom_flagship_manifest.py`, `scripts/run_mom_flagship.py`
- Scaling claims: `scripts/run_scaling_study.py` and `configs/scaling_study.yaml`
- Logit-lens claims: `aom_logit_lens.py` and `scripts/run_logit_lens_dataset.py`

When debugging an extended-analysis discrepancy:

- Check the exported manifest or study config first.
- Check backend alignment and indexing assumptions second.
- Check the originating CLT or evaluation artifact third.
- Only then inspect visualization or summary code.
