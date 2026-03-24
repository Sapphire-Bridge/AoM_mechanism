# SCAN-000

Repository scan for `/Users/felixb/AoM_mechanism_reviewer_fix` on 2026-03-22 at git HEAD `67ca0bd`.

## Repo Snapshot

- Purpose: paper-oriented mechanistic-interpretability repo for AoM / Mechanics of Meaning experiments, with behavioral evaluation, raw activation patching, CLT/SAE patching, comparability analysis, paper verification, and public artifact release.
- Packaging: importable Python package `aom` plus many top-level research scripts.
- Python baseline: `pyproject.toml` requires Python `>=3.10`.
- Test config: `pytest.ini` points at `tests/` and uses `-q`.
- Dependency style: pinned `requirements.txt`, including `torch==2.5.1`, `transformers==4.57.3`, `tokenizers==0.22.1`, `numpy==1.26.4`, `matplotlib==3.9.2`, `PyYAML==6.0.3`, `pytest==9.0.1`.

## Repo Metrics

- Tracked files: about 300.
- Approximate tracked lines: 89,668.
- Approximate tracked Python lines: 52,285.
- On-disk repo size: about 1.9G.
- Largest top-level directories by size:
- `clt_bundles/` about 1.7G
- `results/` about 3.2M
- `aom/` about 2.3M
- `scripts/` about 1.5M
- `data/` about 676K
- `tests/` about 564K
- Largest tracked top-level file groups by count:
- `aom/` 71
- `tests/` 67
- `results/` 48
- `scripts/` 45
- `data/` 12
- `docs/` 6

## Top-Level Map

| Path | Role |
| --- | --- |
| `aom/` | Core package: config, data loading, reproducibility, manifests, metrics, interventions, mechanistic backends |
| `scripts/` | Paper runners, reviewer checks, comparability analysis, figure/table generation, release tooling |
| `paper/` | Paper manuscript source |
| `results/` | Checked-in result CSV/JSON support artifacts |
| `clt_bundles/` | Large CLT model/intervention bundles |
| `data/` | Datasets and manifests |
| `tests/` | Unit and integration tests |
| `docs/` | Project docs, including CLT assumptions and this context pack |

## Paper-Critical Anchor Map

| Path | Why it matters |
| --- | --- |
| `README.md` | Main reproduction narrative and environment expectations |
| `aom_eval.py` | Canonical behavioral evaluation entrypoint and provenance writer |
| `aom/repro.py` | Seed control and determinism policy |
| `aom/run_manifest.py` | Standard run manifest schema and writer |
| `aom/data/loaders.py` | Dataset loading with manifest-aware provenance |
| `aom/provenance/protocol.py` | Frozen protocol provenance and hash enforcement |
| `aom/interventions/activation_patching.py` | Raw hidden-state patching kernels across supported decoder architectures |
| `aom/metrics/disamb.py` | Main DISAMB scoring and raw context-swap patching logic |
| `aom/metrics/clt_cpt.py` | CLT patching and recovery/control analysis |
| `aom/interventions/clt_loader.py` | CLT bundle loading and metadata normalization |
| `aom/interventions/clt_patch.py` | CLT latent replacement policies and reconstruction hooks |
| `aom/interventions/sae_loader.py` | SAE weight/metadata loading |
| `aom/interventions/sae_patching.py` | SAE feature patching kernels |
| `scripts/run_paper.py` | Main smoke/full paper orchestration |
| `scripts/clt_raw_comparability.py` | Raw-vs-CLT comparability and control-arm analysis |
| `scripts/mom_endpoint_decomp_analyze.py` | Endpoint decomposition summaries and markdown report generation |
| `scripts/generate_paper_figures.py` | Figure assembly from stable artifact names |
| `scripts/check_evidence_contract.py` | Evidence-contract consistency checks |
| `scripts/verify_mom_paper.py` | Support-artifact verification against checked-in references |
| `scripts/release_json_artifacts.py` | Sanitized public JSON/CSV release builder |
| `paper/MoM_paper.md` | Claim surface |
| `MoM_evidence_contract.md` | Claim-to-evidence binding surface |

## Hotspots

Largest Python files are concentrated in orchestration and intervention code:

- `scripts/run_paper.py` about 2,816 lines
- `scripts/clt_raw_comparability.py` about 2,008 lines
- `aom/metrics/clt_cpt.py` about 1,881 lines
- `aom_eval.py` about 1,628 lines
- `scripts/clt_raw_comparability_tasklib.py` about 1,487 lines
- `scripts/generate_data.py` about 1,312 lines
- `aom/metrics/disamb.py` about 986 lines
- `mom_circuits_tlens.py` about 931 lines
- `scripts/release_json_artifacts.py` about 897 lines
- `aom/mechanistic/backends/transformer_lens.py` about 853 lines

Interpretation:

- The repo’s real complexity is not in packaging. It is in experiment runners, provenance enforcement, and paper-support analysis scripts.
- `scripts/run_paper.py`, `aom_eval.py`, and the patching modules are the fastest way to understand how claims are generated.
- `scripts/release_json_artifacts.py` is a late-stage bottleneck for publication correctness.

## Current Worktree State

- The worktree is already dirty outside `docs/context/`.
- Existing modifications include `Makefile`, `README.md`, several `scripts/*.py` files, test files, and result artifacts.
- Context generation should avoid touching those files unless explicitly asked.

## Observed Architectural Shape

- Behavioral evaluation and patching share common dataset loaders, config handling, seed control, and manifest/protocol infrastructure.
- Raw activation patching is model-architecture-aware and operates at block-output spans.
- CLT and SAE interventions add learned latent-space transformations on top of the same prompt/task structure.
- Paper-level runners stage datasets, enforce manifests, execute experiment families, then aggregate into verification, figure, and release artifacts.

## Open Edges

- Several wrapper scripts exist beyond the core paths inspected here, but the central paper path is already identifiable from `README.md`, `scripts/run_paper.py`, `scripts/run_mom_paper.py`, and the verification/release scripts.
- Some long-running result directories in `results/` were not re-derived during this scan. Treat checked-in artifacts as inputs to verification tooling, not as independently revalidated outputs.

## Recommended Entry Order

- Start with `docs/context/CONTEXT-003.md` if the task is paper reproduction or runner debugging.
- Start with `docs/context/CONTEXT-002.md` if the task is about patching behavior, intervention correctness, or metric discrepancies.
- Start with `docs/context/CONTEXT-001.md` if the task is provenance, manifests, dataset binding, or reproducibility.
- Start with `docs/context/CONTEXT-004.md` if the task is publication verification, contract linting, or public release.
- Start with `docs/context/CONTEXT-005.md` if the task is flagship tooling, TransformerLens analysis, or scaling-study extensions.
