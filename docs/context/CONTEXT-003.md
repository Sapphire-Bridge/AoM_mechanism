# CONTEXT-003

## Scope

Files covered here:

- `scripts/run_paper.py`
- `scripts/run_mom_paper.py`
- `scripts/run_readme_reproduction.py`
- `scripts/reviewer_quickcheck.py`
- `scripts/clt_raw_comparability.py`
- `scripts/clt_raw_comparability_tasklib.py`
- `scripts/mom_endpoint_decomp_analyze.py`
- `scripts/generate_paper_figures.py`
- `scripts/make_tables.py`
- `docs/CLT_V1.md`

## Architecture Overview

This layer turns the core evaluation and intervention code into a paper workflow.

- `scripts/run_paper.py` is the main orchestrator. It stages datasets, chooses modes, builds commands for experiment families, and collects outputs.
- `scripts/run_mom_paper.py` packages the README-style reproduction path and the paper-support artifact path into a reviewer-facing wrapper.
- `scripts/reviewer_quickcheck.py` is the fast readiness probe for offline smoke, paper-runner dry-run, and local asset/cache sanity.
- `scripts/clt_raw_comparability.py` and `scripts/clt_raw_comparability_tasklib.py` handle the main raw-vs-CLT comparability analysis and control arms.
- `scripts/mom_endpoint_decomp_analyze.py` reduces raw comparability outputs into pair-level and layer-level summaries with bootstrap statistics.
- `scripts/generate_paper_figures.py` and `scripts/make_tables.py` turn those artifacts into paper presentation assets.

## Key Flows

### 1. Canonical paper runner

`scripts/run_paper.py` is the main gateway for both smoke tests and larger paper-like runs.

Important behaviors observed in this file:

- `ensure_paper_dataset()` creates or verifies a deterministic paper dataset manifest and bundle identity.
- Command builders assemble the different experiment families rather than requiring users to remember each raw CLI.
- `run_smoke()` creates a tiny local GPT-2 path and smoke datasets, then exercises behavioral eval and selected mechanistic analyses before summarizing results.
- `_run_m1max_impl()` drives larger staged runs and contains mode-specific logic for machine constraints.
- `run_m1max()` uses the `sdpa` path, while `run_m1max_safe()` switches to eager attention and splits behavioral execution by model.

### 2. README and reviewer-facing orchestration

- `scripts/run_mom_paper.py` wraps the main reproduction flow around support-artifact generation used by the paper.
- It defines fixed support commands for the important raw, CLT, and fixed-layer SAE result families.
- `scripts/run_readme_reproduction.py` provides a smaller reproduction path aligned with the README.
- `scripts/reviewer_quickcheck.py` is the fastest practical starting point when validating a new machine. It checks offline smoke, paper-runner dry-run, and local cache/asset presence.

### 3. CLT vs raw comparability analysis

- `scripts/clt_raw_comparability.py` is the analysis bottleneck for raw-vs-CLT effect alignment.
- It assumes same-site `resid_post` CLT bundles and uses that assumption as part of the validity surface.
- It computes primary log-odds decomposition plus multiple control arms such as PCA, random, and stress-style baselines.
- Inclusion and health flags such as `all_arms_success`, `invariant_all_pass`, and `analysis_included` determine whether a row should contribute to the summary.
- The summary JSON records run configuration, site-equivalence assumptions, counts, and layerwise aggregates.

- `scripts/clt_raw_comparability_tasklib.py` generalizes that comparability pattern across `disamb`, `cf`, and `coh` style tasks.
- It centralizes aggregation and bootstrap helpers so those tasks do not drift apart statistically.

### 4. Endpoint decomposition

- `scripts/mom_endpoint_decomp_analyze.py` converts summary rows into pair-level aggregates and layer summaries.
- It computes bootstrap effect summaries and writes both CSV-style outputs and markdown reporting.
- This is where comparability numbers become more directly legible for manuscript support.

### 5. Figures and tables

- `scripts/generate_paper_figures.py` maps stable result artifact names onto figure outputs.
- It expects specific input CSV and JSON paths under `results/`, especially the overnight mechanistic runs and the comparability summary JSON.
- `scripts/make_tables.py` regenerates paper tables from checked result files.
- `docs/CLT_V1.md` is a useful reference for the intended CLT assumptions behind the same-site comparability path.

## Research Invariants

- The paper dataset manifest should be generated once and reused, not recreated ad hoc per downstream script.
- Smoke mode is not just a convenience. It is the shortest end-to-end validation path for runner correctness.
- Raw-vs-CLT comparability is only interpretable under the site-compatibility assumptions encoded in the runner and `docs/CLT_V1.md`.
- Control arms are part of the analysis definition. Excluding PCA/random/stress comparisons changes the meaning of the headline results.
- Figure and table scripts assume specific artifact schemas and often specific filenames. If upstream outputs are renamed, update these consumers deliberately.

## Interfaces

Main user-facing scripts:

- `scripts/run_paper.py`
- `scripts/run_mom_paper.py`
- `scripts/run_readme_reproduction.py`
- `scripts/reviewer_quickcheck.py`
- `scripts/clt_raw_comparability.py`
- `scripts/mom_endpoint_decomp_analyze.py`
- `scripts/generate_paper_figures.py`
- `scripts/make_tables.py`

Key outputs from this layer:

- behavioral result CSVs
- raw and CLT support CSVs
- comparability summary JSON
- endpoint decomposition reports
- paper figures
- regenerated tables

## Cached Artifacts And Provenance

- These runners assume local model caches and local CLT/SAE assets for offline or reviewer-friendly operation.
- `scripts/reviewer_quickcheck.py` is the best indicator of whether the necessary local assets exist before attempting a long run.
- Figure and table generation mostly consume checked-in or previously generated result artifacts rather than rerunning experiments.

## Operational Notes / Gotchas

- `scripts/run_paper.py` is a large command builder plus orchestrator. When debugging it, first determine whether the failure is dataset staging, command composition, subprocess execution, or result summarization.
- Comparability scripts can look numerically healthy while still failing inclusion invariants. Always inspect the row flags.
- Some figure inputs are intentionally hard-coded to stable result filenames. This is convenient for paper assembly but brittle during refactors.
- The safe M1 Max path exists for a reason. If a run only fails under one attention backend, treat that as a reproducibility issue, not just a performance issue.

## Claim Traceability

- End-to-end paper run logic: `scripts/run_paper.py`
- Reviewer/reproduction wrapper: `scripts/run_mom_paper.py` and `scripts/run_readme_reproduction.py`
- Machine readiness and local cache assumptions: `scripts/reviewer_quickcheck.py`
- Raw-vs-CLT support claims: `scripts/clt_raw_comparability.py` and `scripts/clt_raw_comparability_tasklib.py`
- Endpoint decomposition claims: `scripts/mom_endpoint_decomp_analyze.py`
- Figure/table rendering claims: `scripts/generate_paper_figures.py` and `scripts/make_tables.py`

To trace a paper figure back to code:

- Start at `scripts/generate_paper_figures.py`.
- Identify the exact input artifact path it expects.
- Trace that artifact back to `scripts/clt_raw_comparability.py` or `scripts/run_paper.py`.
- Then follow the underlying intervention code into `CONTEXT-002`.
