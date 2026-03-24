# CONTEXT-001

## Scope

Files covered here:

- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `pytest.ini`
- `aom/config.py`
- `aom/repro.py`
- `aom/models/loader.py`
- `aom/utils.py`
- `aom/run_manifest.py`
- `aom/run_summary.py`
- `aom/data/loaders.py`
- `aom/data/dataset_manifest.py`
- `aom/data/bundle_manifest.py`
- `aom/provenance/protocol.py`

## Architecture Overview

This layer is the repo’s reproducibility spine.

- `aom/repro.py` centralizes seed setting and determinism policy through `ReproConfig`, `seed_everything()`, `collect_versions()`, and git/runtime capture.
- `aom/models/loader.py` standardizes model and tokenizer loading, including revision handling, local cache resolution, and commit-hash recording.
- `aom/data/loaders.py` is the single place where paper datasets enter the system with manifest-aware metadata.
- `aom/data/dataset_manifest.py` and `aom/data/bundle_manifest.py` turn raw files into stable hashed identities.
- `aom/provenance/protocol.py` binds runs to frozen protocol hashes so evaluation rules cannot drift silently.
- `aom/run_manifest.py` and `aom/run_summary.py` turn execution metadata into machine-checkable run outputs.
- `aom/utils.py` carries shared scoring and device helpers used by evaluation code.

The result is that most higher-level scripts do not invent their own provenance logic. They compose this layer.

## Key Flows

### 1. Config and environment setup

- `pyproject.toml` defines the package baseline and declares `aom` as the packaged module.
- `requirements.txt` pins the runtime stack tightly. This matters because model, tokenizer, and activation semantics are version-sensitive.
- `aom/config.py` loads YAML/structured config, validates keys, and resolves relative paths so downstream scripts can accept config files without duplicating path logic.
- `pytest.ini` restricts test discovery to `tests/`, which is useful when adding new script helpers that should not accidentally be picked up as tests.

### 2. Run reproducibility and determinism

- `aom/repro.py` seeds Python `random`, NumPy, PyTorch, and CUDA through one path.
- Determinism is explicit rather than implied. The code distinguishes modes such as `strict`, `best_effort`, and `off`.
- Version capture is part of the run surface, not an afterthought. Git hash, library versions, and relevant CUDA/CUBLAS state are collected up front.

### 3. Model and tokenizer provenance

- `aom/models/loader.py` loads causal LMs with explicit model revision and tokenizer revision handling.
- Under local/offline execution it can resolve Hugging Face cache snapshots rather than assuming network access.
- The loader records effective provenance such as model commit hash and effective tokenizer revision so later CSV rows and manifests remain attributable.

### 4. Dataset identity and bundle identity

- `aom/data/loaders.py` exposes task-specific loaders such as `load_disamb_pairs_with_manifest()`, `load_counterfactual_pairs_with_manifest()`, `load_coherence_items_with_manifest()`, and `load_authority_pairs_with_manifest()`.
- `aom/data/dataset_manifest.py` computes hashed dataset metadata through `build_dataset_manifest()` and `sha256_file()`.
- `aom/data/bundle_manifest.py` derives a stable bundle identifier with `compute_bundle_id()` and validates that multi-file dataset bundles are internally consistent.

### 5. Frozen protocol enforcement

- `aom/provenance/protocol.py` resolves expected protocol provenance and can enforce that the active protocol sources match their recorded hashes.
- This prevents quiet changes to scoring, rendering, or intervention logic from being mistaken for the same experiment.

### 6. Run manifest and summary output

- `aom/run_manifest.py` assembles structured run metadata with helpers such as `build_run_manifest()`, `redact_argv()`, `write_run_manifest()`, and `validate_run_manifest()`.
- `aom/run_summary.py` compresses failures, skips, invalid counts, and overall status into a simple PASS/WARN/FAIL surface for higher-level scripts.

## Research Invariants

- Always seed through `aom/repro.py` or an equivalent path that updates the same provenance fields. Ad hoc seed calls are insufficient.
- Dataset comparisons are only meaningful when the same dataset manifest or derived bundle ID is preserved across runs.
- Protocol-bound experiments should enforce hashes through `aom/provenance/protocol.py`. If a script bypasses that layer, it weakens claim traceability.
- Model identity is more than the repo name. Revision and tokenizer provenance must stay attached to outputs.
- Manifests are first-class artifacts. If a tool writes a CSV or JSON without a sidecar manifest or equivalent metadata, it is probably below repo standard.

## Interfaces

Key imported surfaces:

- `ReproConfig`, `seed_everything()`, `collect_versions()`
- `load_causal_lm()`
- `load_*_with_manifest()` dataset loaders
- `build_dataset_manifest()`, `compute_bundle_id()`, `validate_bundle_manifest()`
- `resolve_protocol_provenance()`, `enforce_protocol_bindings()`
- `build_run_manifest()`, `write_run_manifest()`, `validate_run_manifest()`
- `RunSummary`

Operational entrypoints that depend heavily on this layer:

- `aom_eval.py`
- `scripts/run_paper.py`
- `scripts/run_mom_paper.py`
- `scripts/run_scaling_study.py`
- `scripts/run_mom_flagship.py`

## Cached Artifacts And Provenance

- Hugging Face model snapshots and tokenizer revisions may be consumed from local cache under offline settings.
- Dataset manifests are stable records of file hashes and should travel with generated data.
- Bundle manifests are the correct identity surface for multi-file paper datasets.
- Run manifests should be treated as the authoritative explanation of how a result file was produced.

## Operational Notes / Gotchas

- This repo mixes importable package code with many top-level scripts. Do not assume the package boundary tells you which code is publication-critical.
- A dirty worktree matters for provenance. Future runs should rely on manifest metadata, not just local file timestamps.
- The package list in `pyproject.toml` only includes `aom`. Many important scripts live at repo root and will not be captured by package-only audits.
- `aom/utils.py` contains shared scoring behavior and device selection. Small changes there can have large downstream behavioral effects.

## Claim Traceability

- Reproducibility claim: `README.md` plus `aom/repro.py` plus `aom/run_manifest.py`
- Dataset identity claim: `aom/data/loaders.py` plus `aom/data/dataset_manifest.py` plus `aom/data/bundle_manifest.py`
- Protocol-freeze claim: `aom/provenance/protocol.py`
- Model provenance claim: `aom/models/loader.py`
- Run health/status claim: `aom/run_summary.py`

When debugging a suspicious result, the usual order is:

- Check the run manifest.
- Confirm dataset manifest and bundle ID.
- Confirm protocol hashes.
- Confirm seed and runtime versions.
- Only then inspect the metric or intervention code.
