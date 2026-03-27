# AoM_mechanism

Paper-grade artifact and verification repository for the mechanics-focused AoM / Mechanics of Meaning (MoM) analyses, comparability runs, and reproduction paths.

Reviewer guidance: see `PAPER_VERIFICATION_GUIDE.md`. AI-agent routing: see `CONTEXT.md`. Repo-specific working rules for agents: see `AGENTS.md`.

## Paper

- Preprint: https://zenodo.org/records/18906800

- AoM-DISAMB: context-sensitive disambiguation via minimal pairs
- AoM-CF: minimal-pair intervention sensitivity via directional preference shift
- AoM-COH: discourse-level coherence / constraint tracking
- CPT: context-swap activation patching on internal states
- SAE-CPT: SAE feature-space context-swap patching on DISAMB
- CLT-CPT: CLT latent-space context-swap patching on DISAMB

The paper-facing mechanism results are maintained under the CLT-labeled comparability and endpoint-decomposition pipelines in `scripts/clt_raw_comparability*.py` and `scripts/mom_endpoint_decomp_analyze.py`.

## Repository status

- `results/` contains sanitized paper-facing reference artifacts for this public release, not the full internal historical results tree.
- `public_artifacts/` contains `RELEASE_MANIFEST.json` plus portability and audit reports emitted by `scripts/release_json_artifacts.py publish`.
- The fastest clean-clone validation path is `make check`.
- The reviewer-facing full MoM reproduction command is `make reproduction`.
- The reviewer-facing full accelerator sweep command is `make paper-reproduction-gpu`.
- The full Gemma-2-2B MoM reproduction is a heavyweight multi-hour verification path that assumes local model assets plus the CLT and SAE artifacts cited below.

## Start Here

If you are evaluating this repo as an engineer, interviewer, or lab reviewer, use the shortest path that answers your question. These are rough time budgets on a machine with dependencies installed; a fresh connected machine should run `make reviewer-assets` once before the reviewer-only commands below.

### About 3 minutes: repo boots, tests pass, smoke path works

```bash
make check
```

Use this when you want the fastest high-signal check that the repo is wired correctly. It runs the offline test suite, evidence-contract checks, and the canonical smoke runner.

### About 20 minutes: reviewer readiness plus one real result

```bash
make reviewer-assets
make reviewer-check
make one-result-check-gpu
```

If no accelerator is available, use:

```bash
make reviewer-assets
make reviewer-check
make one-result-check
```

This is the best interview or lab-demo path on a fresh connected machine. `make reviewer-assets` downloads the pinned reviewer assets once. `make reviewer-check` then confirms the environment, offline smoke route, paper-runner dry-run, and local assets. `make one-result-check-gpu` runs a substantive accelerator-backed verification against the tracked public reference. On CPU-only machines, `make one-result-check` provides the same claim-level check without the accelerator requirement.

### Full run: strict paper reproduction, plus optional accelerator sweep

```bash
make reproduction MOM_PAPER_ARGS="--run_root /tmp/mom_paper_review_run"
```

Optional CUDA/MPS showcase:

```bash
make paper-reproduction-gpu PAPER_GPU_ARGS="--results_dir /tmp/paper_cuda_validated --local_files_only"
```

`make reproduction` is the canonical strict paper-proof path. It is intentionally CPU-pinned for the core comparability and support stages. `make paper-reproduction-gpu` is the broad accelerator sweep for showcasing CUDA/MPS operability; it is not the canonical claim-verification path.

## One-command check

```bash
make check
```

`make check` creates a local `.venv` with `python3.11` when available (otherwise `python3`), installs `requirements.txt`, and runs the lightweight canonical verification path:

- `python -m pytest -q`
- `python scripts/check_evidence_contract.py`
- `python scripts/check_evidence_contract_fields.py`
- `python scripts/run_paper.py smoke`

If you prefer to manage the environment manually, the equivalent commands are:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python scripts/check_evidence_contract.py
python scripts/check_evidence_contract_fields.py
python scripts/run_paper.py smoke
```

## One-command MoM paper run

```bash
make reproduction
```

`make reproduction` creates the local `.venv` if needed, then runs the paper-facing MoM package into a fresh temp directory outside the repo:

- the core CLT comparability / endpoint-decomposition path verified against the checked-in reference artifacts
- the six-layer raw vs CLT layer-profile support run used in §4
- the fixed-layer SAE specificity support run cited in §4 / Appendix A.2

For cache-only / offline-style execution, use:

```bash
make reproduction
```

`make mom-paper` remains as a backward-compatible legacy target name if you need the older command spelling, but it does not force `--local_files_only`.

## Repository layout

- `aom/`: library code for datasets, metrics, interventions, model loading, and reporting
- Root `aom_*.py` files: stable CLI entry points kept at the repository root for compatibility
- `scripts/`: orchestration, release tooling, paper runners, and analysis utilities
- `tests/`: offline deterministic test suite
- `data/`: baseline datasets for smoke tests and local evaluation examples
- `data_paper_hardened_v2/`: paper-specific hardened dataset bundle retained for artifact compatibility
- `results/`: sanitized reference artifacts included for immediate inspection
- `public_artifacts/`: release manifest plus audit and portability reports
- `paper/MoM_paper.md` and `MoM_evidence_contract.md`: canonical manuscript and evidence contract

## Canonical manuscript checks (MoM)

The canonical paper / evidence pair for repository-level verification is:

- `paper/MoM_paper.md`
- `MoM_evidence_contract.md`

These checks validate the evidence contract table, inline Evidence IDs, and Appendix A.2 claim-to-artifact crosswalk. `check_evidence_contract.py` fails if the manuscript has neither inline IDs nor a populated A.2 crosswalk, if an A.2 row cites an unknown Evidence ID, or if an A.2 artifact path is missing.

```bash
python scripts/check_evidence_contract.py
python scripts/check_evidence_contract_fields.py
python -m pytest -q tests/test_evidence_contract_ids.py tests/test_evidence_contract_fields.py
```

## Results and publication artifacts

This public repository carries a curated sanitized release surface:

- `results/` contains the paper-facing CSV / JSON reference artifacts included with this release
- `public_artifacts/RELEASE_MANIFEST.json` records the publication-copy file set
- `public_artifacts/audit/` contains JSON / CSV audit reports and portability scan output
- `figures/` contains the paper figures referenced from `paper/MoM_paper.md`

Appendix A.2 in the paper maps main-text and appendix claims to their backing artifact paths.

## Entry points

Primary entry points:

- `aom_eval.py`: behavioral AoM evaluation plus activation / SAE / CLT patching
- `aom_mechanistic.py`: induction-head measurement runner
- `aom_logit_lens.py`: logit-lens runner
- `aom_sae_check.py`: SAE sterility / round-trip check
- `aom_sae_sweep.py`: SAE threshold sweep runner

Additional specialized entry points remain at the repository root for compatibility, while orchestration and release helpers live under `scripts/`.

## Quickstart examples

Generate a larger templated dataset:

```bash
python scripts/generate_data.py --out_dir data --seed 0
```

Reproduce the original cue-heavy DISAMB templates:

```bash
python scripts/generate_data.py --out_dir data --seed 0 --disamb_mode easy
```

Include CF sham controls and COH ablation controls:

```bash
python scripts/generate_data.py --out_dir data --seed 0 --cf_include_shams --coh_include_controls
```

Run an evaluation on a local HuggingFace model:

```bash
python aom_eval.py \
  --model_name_or_path gpt2 \
  --local_files_only \
  --disamb_path data/disamb_pairs.jsonl \
  --cf_path data/counterfactual.jsonl \
  --coh_path data/coherence.jsonl \
  --run_patching \
  --csv_path results/aom_results.csv
```

Qwen example (downloads unless `--local_files_only` is set):

```bash
python aom_eval.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --torch_dtype bfloat16 \
  --attn_implementation eager \
  --csv_path results/qwen_behavioral.csv
```

Sweep multiple models into one CSV:

```bash
python aom_eval.py \
  --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B \
  --device mps \
  --torch_dtype float32 \
  --attn_implementation eager \
  --csv_path results/qwen_behavioral.csv
```

Notes:
- Metrics are log-prob based and deterministic in `model.eval()`, with uncertainty estimated by bootstrap over items.
- Log-probs are computed via `log_softmax` in `--logprobs_dtype` (default `float32`) for numeric stability, and `--strict_finite` fails fast on non-finite values.
- `scripts/generate_data.py` can generate sham CF controls and coherence ablation controls; see `data/README.md`.
- For publication-grade extensions, expand the datasets in `data/` and report confidence intervals explicitly.

## Supplementary tools (not covered in paper body)

These entry points support related research infrastructure but are not referenced directly in the MoM paper body:

| Script | Purpose |
|---|---|
| `aom_authority_game.py` | Authority language-game CPT + SAE recovery |
| `aom_completeness.py` | Completeness / dark-matter metrics |
| `aom_disamb_path_decomp.py` | DISAMB path decomposition (TransformerLens) |
| `aom_why_fetch.py` | Why-fetch QK/OV analysis |
| `aom_clt_check.py` | CLT preflight: calibration / reconstruction / identity-drift |
| `mom_circuits_tlens.py` | Head-level circuit analysis (TransformerLens) |
| `scripts/run_mom_flagship.py` | Multi-model flagship runner |
| `scripts/run_safety_chat.py` | Safety chat conversion / eval orchestration |

## SAE / CLT patching in `aom_eval.py`

`aom_eval.py` can run SAE and CLT patching in the same row as behavioral AoM metrics.

Canonical MoM comparability/decomposition path (paper-facing):
- DISAMB comparability: `scripts/clt_raw_comparability.py`
- CF/COH task-axis comparability: `scripts/clt_raw_comparability_cf.py`, `scripts/clt_raw_comparability_coh.py`
- Endpoint decomposition: `scripts/mom_endpoint_decomp_analyze.py`
- CLT v1 site contract is same-site only: `encode_site == decode_site == writeback_site == resid_post` (see `docs/CLT_V1.md`)

Reviewer-safe reproduction commands for the core MoM claims:

These commands write into a temp directory outside the repo so the checked-in release artifacts under `results/` stay untouched and `git status` remains clean. The main-text DISAMB runs below intentionally use `data/disamb_pairs.jsonl`; the fixed-layer specificity appendix artifact uses `data_paper_hardened_v2/disamb_pairs.jsonl` and is reported separately rather than pooled with the six-layer table. The canonical multi-layer CLT bundle path is `clt_bundles/gemma-scope-2b-pt-res_sweep_smoke`; the `_smoke` suffix is historical, but this bundle contains the release layers `4/8/12/16/20/24`.

Fresh connected reviewer machine:

```bash
make reviewer-assets
```

This downloads the pinned `google/gemma-2-2b` snapshot, materializes the pinned CLT reviewer bundle, and caches the fixed-layer SAE support files needed by the strict paper path. Run this once per machine before the offline reviewer commands below.

Shortest reviewer readiness check after assets are present:

```bash
make reviewer-check
```

This is the best command for an already-prepared reviewer setup. It may bootstrap the virtualenv/dependencies first, then runs three fast gates before any multi-hour job:
- an offline CPU smoke run with a tiny local model,
- a dry-run of the paper runner with the reviewer-safe CPU settings,
- a local asset check for `google/gemma-2-2b`, the CLT bundle/source cache, and the fixed-layer SAE support files.

If it ends with `ready_for_offline_paper_run: PASS`, the reviewer can immediately choose either the fast single-result check or the full paper run using the exact commands printed at the end of the check. If assets are missing, the quickcheck now points back to `make reviewer-assets`.

For an already-installed environment, CI, or a repeated offline readiness check that should avoid `make` bootstrap behavior, run the wrapper directly:

```bash
python scripts/reviewer_quickcheck.py --cleanup
```

The quickcheck validates execution readiness plus offline asset readiness. It intentionally does not regenerate the hardened paper dataset during the smoke step; that path uses `--skip_dataset` so the repo stays clean.
If a reviewer machine is unusually slow, the smoke and dry-run timeouts can be raised with `--smoke-timeout-seconds`, `--paper-dry-run-timeout-seconds`, or the env vars `MOM_REVIEWER_SMOKE_TIMEOUT_SECONDS` and `MOM_REVIEWER_DRY_RUN_TIMEOUT_SECONDS`.

Fastest one-command single-claim verification:

```bash
make one-result-check
```

This runs a real layer-4 DISAMB controls verification against the tracked public reference and writes:
- a compact markdown report
- a machine-readable command/check log
- the layer-4 controls CSV and summary JSON

The quick-result report ends with `overall_status: PASS` or `overall_status: FAIL`.

Accelerator variant of the same single-result check:

```bash
make one-result-check-gpu
```

This runs the same layer-4 controls verification on the best available accelerator:
- CUDA if available
- otherwise MPS if available
- otherwise it fails clearly instead of silently falling back to CPU

It still verifies against the same tracked public reference as `make one-result-check`.
On non-CPU devices, small drift in the auxiliary PCA baseline is reported as `WARN` rather than `FAIL`; the main ordering and core control checks remain pass/fail. Treat this GPU path as an accelerator consistency check, not the canonical paper-proof path.

Reviewer-facing full accelerator sweep:

```bash
make paper-reproduction-gpu
```

This convenience command chooses the correct broad accelerator preset automatically:
- `scripts/run_paper.py m1max_safe` on Apple Silicon / MPS
- `scripts/run_paper.py cuda_validated` on CUDA

Use this for the multi-model accelerator sweep. On Apple Silicon, the broad behavioral stage runs conservatively with `eager`, and per-model failures/timeouts are reported clearly instead of hanging silently. The canonical strict paper-proof path remains `make reproduction`.

Fastest one-command full reviewer path:

```bash
make reproduction MOM_PAPER_ARGS="--run_root /tmp/mom_paper_review_run"
```

Notes for the one-command runner:
- `RUN_ROOT` must start empty.
- The runner pins both the strict paper stages and the paper-support stages to CPU to avoid Apple Silicon / MPS drift.
- It writes `$RUN_ROOT/mom_paper_reproduction_report.md` and `$RUN_ROOT/mom_paper_reproduction_log.json`.
- A successful run reports `overall_status: PASS`.
- Standalone verifier: `python scripts/verify_mom_paper.py --run_root "$RUN_ROOT"`.
- `make mom-paper` remains as a backward-compatible legacy target name, but it does not force `--local_files_only`; `make reproduction` is the reviewer-facing target.

Pod / RunPod launch helpers:

These are thin wrappers around the existing repo commands. They activate the repo venv, set HF/cache env vars, launch under `nohup`, and write sibling `.log` and `.pid` files.

```bash
bash scripts/pod_run_one_result_gpu.sh
bash scripts/pod_run_all_results_gpu.sh
bash scripts/pod_run_paper_cpu.sh
```

Intent:
- `pod_run_one_result_gpu.sh`: quick accelerator claim check
- `pod_run_all_results_gpu.sh`: full accelerator sweep via `scripts/run_paper_accelerated.py`
- `pod_run_paper_cpu.sh`: canonical strict paper reproduction via `scripts/run_mom_paper.py`

Notes:
- `LOCAL_FILES_ONLY=1` is a strict boolean toggle; anything else leaves downloads enabled.
- `HF_HOME` is the primary cache root; `TRANSFORMERS_CACHE` is set only for backward compatibility.
- On success, each wrapper creates a sibling `*.tar.gz` archive, a `*.sha256` checksum, and a simple `*.status` file before any pod stop action.
- Default post-success behavior is `RUNPOD_POST_SUCCESS_ACTION=auto`: on RunPod, if `RUNPOD_POD_ID` and `runpodctl` are available, the wrapper stops the pod after archiving so GPU billing ends while the archived outputs remain on the workspace volume. Use `RUNPOD_POST_SUCCESS_ACTION=terminate` only if you explicitly want full pod deletion.
- The `.pid` file is removed by the post-success wrapper when the background job exits; trust the `.status` file for final state.
- `pod_run_paper_cpu.sh` is intentionally CPU-pinned because `scripts/run_mom_paper.py` uses CPU for both the core comparability path and the support path.
- Fresh pods may still require `huggingface-cli login` or `HF_TOKEN` plus accepted model licenses:
  - `google/gemma-2-2b` for the one-result and paper runs
  - `meta-llama/*` models when the accelerator sweep resolves to CUDA / `cuda_validated`
- Use `LOCAL_FILES_ONLY=1` only when the required caches are already present.

```bash
RUN_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mom_core.XXXXXX")"
echo "$RUN_ROOT"

# 1) DISAMB matched controls (PCA + random orthogonal + recon/residual stress arms)
python scripts/clt_raw_comparability.py \
  --model_name_or_path google/gemma-2-2b \
  --disamb_path data/disamb_pairs.jsonl \
  --clt_repo clt_bundles/gemma-scope-2b-pt-res_sweep_smoke \
  --layers 4,8,12 \
  --device cpu \
  --torch_dtype float32 \
  --seed 42 \
  --bootstrap_n 1000 \
  --bootstrap_seed 42 \
  --no-hard_fail_primary_logodds \
  --primary_logodds_residual_tol 5e-06 \
  --run_pca_baseline \
  --run_random_orth_baseline \
  --run_faithfulness_decomposition_arms \
  --out_csv "$RUN_ROOT/clt_raw_comparability_l4_l8_l12_controls_full.csv" \
  --out_json "$RUN_ROOT/clt_raw_comparability_l4_l8_l12_controls_full.summary.json"

# 2) Strict CPU endpoint reruns used for the main-text mechanism claims
python scripts/clt_raw_comparability.py \
  --model_name_or_path google/gemma-2-2b \
  --disamb_path data/disamb_pairs.jsonl \
  --clt_repo clt_bundles/gemma-scope-2b-pt-res_sweep_smoke \
  --layers 4,8,12 \
  --device cpu \
  --torch_dtype float32 \
  --seed 42 \
  --bootstrap_n 5000 \
  --bootstrap_seed 42 \
  --no-hard_fail_primary_logodds \
  --primary_logodds_residual_tol 5e-06 \
  --out_csv "$RUN_ROOT/r1_full_cpu_f32.csv" \
  --out_json "$RUN_ROOT/r1_full_cpu_f32.summary.json"

python scripts/clt_raw_comparability.py \
  --model_name_or_path google/gemma-2-2b \
  --disamb_path data/disamb_pairs.jsonl \
  --clt_repo clt_bundles/gemma-scope-2b-pt-res_sweep_smoke \
  --layers 4,8,12 \
  --device cpu \
  --torch_dtype float64 \
  --seed 42 \
  --bootstrap_n 5000 \
  --bootstrap_seed 42 \
  --no-hard_fail_primary_logodds \
  --primary_logodds_residual_tol 5e-06 \
  --out_csv "$RUN_ROOT/r1_full_cpu_f64.csv" \
  --out_json "$RUN_ROOT/r1_full_cpu_f64.summary.json"

# 3) Endpoint decomposition from the strict FP64 rerun
python scripts/mom_endpoint_decomp_analyze.py \
  --comparability_csv "$RUN_ROOT/r1_full_cpu_f64.csv" \
  --comparability_summary "$RUN_ROOT/r1_full_cpu_f64.summary.json" \
  --disamb_path data/disamb_pairs.jsonl \
  --tokenizer_name_or_path google/gemma-2-2b \
  --bootstrap_n 5000 \
  --ci 0.95 \
  --seed 42 \
  --primary_residual_tol 5e-06 \
  --out_pair_csv "$RUN_ROOT/r1_full_cpu_f64.endpoint_pair_aggregates_v2.csv" \
  --out_json "$RUN_ROOT/r1_full_cpu_f64.endpoint_decomp_summary_v2.json" \
  --out_md "$RUN_ROOT/r1_full_cpu_f64.endpoint_decomp_summary_v2.md"

# 4) Task-axis comparability side runs (CF / COH)
python scripts/clt_raw_comparability_cf.py \
  --model_name_or_path google/gemma-2-2b \
  --cf_path data/counterfactual.jsonl \
  --coh_path data/coherence.jsonl \
  --clt_repo clt_bundles/gemma-scope-2b-pt-res_sweep_smoke \
  --layers 4,8,12 \
  --device cpu \
  --torch_dtype float32 \
  --seed 42 \
  --bootstrap_n 1000 \
  --bootstrap_seed 42 \
  --out_csv "$RUN_ROOT/clt_raw_comparability_cf_l4_l8_l12_final_f32.csv" \
  --out_json "$RUN_ROOT/clt_raw_comparability_cf_l4_l8_l12_final_f32.summary.json"

python scripts/clt_raw_comparability_coh.py \
  --model_name_or_path google/gemma-2-2b \
  --cf_path data/counterfactual.jsonl \
  --coh_path data/coherence.jsonl \
  --clt_repo clt_bundles/gemma-scope-2b-pt-res_sweep_smoke \
  --layers 4,8,12 \
  --device cpu \
  --torch_dtype float32 \
  --seed 42 \
  --bootstrap_n 1000 \
  --bootstrap_seed 42 \
  --out_csv "$RUN_ROOT/clt_raw_comparability_coh_l4_l8_l12_final_f32.csv" \
  --out_json "$RUN_ROOT/clt_raw_comparability_coh_l4_l8_l12_final_f32.summary.json"
```

Expected generated outputs:
- `$RUN_ROOT/clt_raw_comparability_l4_l8_l12_controls_full.csv`
- `$RUN_ROOT/clt_raw_comparability_l4_l8_l12_controls_full.summary.json`
- `$RUN_ROOT/r1_full_cpu_f32.csv`
- `$RUN_ROOT/r1_full_cpu_f32.summary.json`
- `$RUN_ROOT/r1_full_cpu_f64.csv`
- `$RUN_ROOT/r1_full_cpu_f64.summary.json`
- `$RUN_ROOT/r1_full_cpu_f64.endpoint_decomp_summary_v2.json`
- `$RUN_ROOT/clt_raw_comparability_cf_l4_l8_l12_final_f32.csv`
- `$RUN_ROOT/clt_raw_comparability_cf_l4_l8_l12_final_f32.summary.json`
- `$RUN_ROOT/clt_raw_comparability_coh_l4_l8_l12_final_f32.csv`
- `$RUN_ROOT/clt_raw_comparability_coh_l4_l8_l12_final_f32.summary.json`

Numeric checks to treat as a successful reproduction:
- `$RUN_ROOT/r1_full_cpu_f64.summary.json` should report `counts.n_rows_analysis_included=312`, `counts.n_invariant_fail_rows=0`, and `run_config.bootstrap_n=5000`. Compare against `results/mom_endpoint_plan/r1_full_cpu_f64.summary.json`.
- `$RUN_ROOT/r1_full_cpu_f64.endpoint_decomp_summary_v2.json` should report `counts.n_rows_primary_logodds_applicable=288`, `results_by_layer.{4,8,12}.primary_applicable_pairs.ddm.mean ≈ {0.109087, 0.003583, -0.071915}`, and `results_by_layer.{4,8,12}.primary_applicable_pairs.d_ca_diag_logz.mean = 0.0`. Compare against `results/mom_endpoint_plan/r1_full_cpu_f64.endpoint_decomp_summary_v2.json`.
- `$RUN_ROOT/clt_raw_comparability_l4_l8_l12_controls_full.summary.json` should report `counts.n_rows_analysis_included=312`, `counts.n_pairs_analysis_included=52`, zero invariant failures, and all optional-arm success flags true. At layer 4, the mean ordering should remain `RECON (0.3359) > Raw A (0.2612) > PCA (0.2127) >> Random mean (0.0198)` with `RESID (-0.0552)` mean-negative. Compare against `results/clt_raw_comparability_l4_l8_l12_controls_full.summary.json`.
- `$RUN_ROOT/clt_raw_comparability_cf_l4_l8_l12_final_f32.summary.json` should report `60/60` included rows, `0` invariant failures, and CRR means `0.922/0.915/0.917` for layers `4/8/12`. Compare against `results/clt_raw_comparability_cf_l4_l8_l12_final_f32.summary.json`.
- `$RUN_ROOT/clt_raw_comparability_coh_l4_l8_l12_final_f32.summary.json` should report `480/480` included rows, `0` invariant failures, and CRR means `0.870/0.869/0.855` for layers `4/8/12`. Compare against `results/clt_raw_comparability_coh_l4_l8_l12_final_f32.summary.json`.
- One-command runner: `python scripts/run_mom_paper.py --run_root "$RUN_ROOT" --local_files_only`.
- Scripted verifier for the one-command runner: `python scripts/verify_mom_paper.py --run_root "$RUN_ROOT"`.
- Scripted verifier: `python scripts/verify_readme_reproduction.py --run_root "$RUN_ROOT"`
- Scripted end-to-end runner/report: `python scripts/run_readme_reproduction.py --local_files_only --report_path reports/readme_reproduction_report.md`

Hardware / runtime notes for reviewers:
- `python scripts/run_paper.py smoke` is the CPU-only offline sanity check; use it to verify the environment before any heavy run.
- The commands above assume local access to `google/gemma-2-2b` weights and the CLT bundle path `clt_bundles/gemma-scope-2b-pt-res_sweep_smoke` (materialize it first with `scripts/gemma_scope_to_clt.py --preset readme_core_bundle --local_files_only` if needed).
- The core comparability steps are pinned to CPU in the paper runner to avoid Apple Silicon / MPS invariant drift in the A≈B gate.
- The strict endpoint claims in the paper use CPU reruns (`--device cpu`), not the relaxed MPS workflow. Treat them as multi-hour CPU jobs.
- The checked-in six-layer overnight manifests on the original machine record `wall_time_sec=5091.42` for the raw run and `wall_time_sec=7020.59` for the CLT run (about 85 min and 117 min, respectively).
- `scripts/run_paper.py m1max` and `scripts/run_paper.py cuda_validated` remain the broader preset runners for multi-model paper sweeps. The commands above are the narrower verifier path for the core Gemma 2 2B MoM claims.
- Structural archival gate: `python scripts/archive_readiness_check.py` (or `make archival-check`) runs evidence checks, publication-copy generation, portability scans, bundle dry-runs, and README verification if you supply an existing `--readme_run_root`.

SAE feature-space CPT example:

```bash
python aom_eval.py \
  --model_name_or_path google/gemma-2-2b \
  --run_sae_patching \
  --sae_repo google/gemma-scope-2b-pt-res \
  --sae_width 16k \
  --sae_layers 12 \
  --sae_run_name average_l0_176 \
  --sae_decode_strategy delta_1decode \
  --csv_path results/gemma2b_sae_cpt_layer12.csv
```

CLT latent-space CPT example (local CLT bundle):

```bash
python aom_eval.py \
  --model_name_or_path google/gemma-2-2b \
  --run_clt_patching \
  --clt_repo clt_bundles/gemma-scope-2b-pt-res \
  --clt_width 16k \
  --clt_layers 12 \
  --clt_run_name average_l0_176 \
  --clt_decode_strategy safe_2decode \
  --csv_path results/gemma2b_clt_layer12.csv
```

Guardrails:
- `--run_sae_patching` requires explicit `--sae_repo` and `--sae_layers`.
- `--run_clt_patching` requires explicit `--clt_repo` and `--clt_layers`.
- For HF repos with multiple runs, pass `--sae_run_name/--sae_l0_target` or `--clt_run_name/--clt_l0_target`.
- Patching modes do not support `--device_map` (`--run_patching`, `--run_sae_patching`, `--run_clt_patching`, `--run_patching_specificity`).

## Latest Gemma-2-2B-IT snapshot (2026-02-27)

Current IT artifacts in this repo:

- Behavioral baseline (`2026-02-26T08:07:52Z`): `results/safety_it_20260226T074644Z/gemma2b_it_safety_baseline_seed42.manifest.json`
  - `disamb_accuracy=0.425` (95% CI `[0.360, 0.485]`)
  - `cf_label_accuracy=0.3833` (95% CI `[0.3083, 0.4500]`)
  - `coh_constraint_accuracy=0.850` (95% CI `[0.725, 0.950]`)
  - `aom_composite=0.5972`
- CLT CPT layer sweep (`2026-02-26T09:29:54Z`): `results/safety_it_20260226T074644Z/gemma2b_it_clt_6layer_safety_seed42.manifest.json`
  - `clt_layers=4,8,12,16,20,24`
  - `clt_cpt_mean_max_effect=0.13246` (95% CI `[0.11493, 0.15107]`)
  - `clt_cpt_mean_sham_max_effect=6.37e-06` (95% CI `[2.39e-06, 1.32e-05]`)
  - `clt_cpt_mean_identity_max_abs_effect=0.0` (95% CI `[0.0, 0.0]`)
  - `clt_cpt_flip_rate_at_best_layer=0.075`, `n_directions_total=200`, `n_directions_patched=200`, `n_directions_skipped_misaligned=0`
- CLT top-k/control analysis (`2026-02-27T02:29:27Z`): `results/mom_option_c_bigrun_fp16_google_gemma-2-2b-it/topk.summary.json`
  - Layer 12, `k=20`: top-k `0.0940` (95% CI `[-0.0326, 0.2344]`), random-k `0.0032` (95% CI `[-0.0024, 0.0091]`), bottom-k `0.00070` (95% CI `[0.0, 0.0028]`)
  - Interpretation: for this IT run, top-k direction is positive but not CI-separated from 0 at `k=20`; controls are near zero.

## SAE / CLT preflight CLIs

List available SAE/CLT runs for a layer:

```bash
python aom_sae_check.py \
  --model_name_or_path google/gemma-2-2b \
  --sae_repo google/gemma-scope-2b-pt-res \
  --layer 12 \
  --width 16k \
  --list_runs

python aom_clt_check.py \
  --model_name_or_path google/gemma-2-2b \
  --clt_repo clt_bundles/gemma-scope-2b-pt-res \
  --layer 12 \
  --width 16k \
  --list_runs
```

Run SAE sterility (+ optional calibration):

```bash
python aom_sae_check.py \
  --model_name_or_path google/gemma-2-2b \
  --sae_repo google/gemma-scope-2b-pt-res \
  --layer 12 \
  --width 16k \
  --run_name average_l0_176 \
  --calibrate \
  --out_json results/sae_check_layer12.json
```

Run CLT preflight (identity drift + recon telemetry + optional calibration):

```bash
python aom_clt_check.py \
  --model_name_or_path google/gemma-2-2b \
  --clt_repo clt_bundles/gemma-scope-2b-pt-res \
  --layer 12 \
  --width 16k \
  --run_name average_l0_176 \
  --calibrate \
  --decode_strategy safe_2decode \
  --out_json results/clt_check_layer12.json
```

Run CLT top-k recovery (selection/evaluation split, concentration + recovery curves):

```bash
python aom_clt_topk_recovery.py --help
```

Run CLT feature interpretation (activation profiles, selectivity, family clustering):

```bash
python aom_clt_feature_analysis.py --help
```

Run CLT head attribution in feature space (+ top-vs-random ablation validation):

```bash
python aom_clt_head_attribution.py --help
```

Run SAE threshold sweep:

```bash
python aom_sae_sweep.py \
  --model_name_or_path google/gemma-2-2b \
  --sae_repo google/gemma-scope-2b-pt-res \
  --layer 12 \
  --width 16k \
  --run_name average_l0_176 \
  --thresholds 0.0,0.1,0.5,1.0,2.0 \
  --out_csv results/sae_sweep_layer12.csv \
  --out_json results/sae_sweep_layer12.json
```

Prepare CLT bundles from Gemma Scope exports:

```bash
python scripts/gemma_scope_to_clt.py \
  --layers 12,18 \
  --width 16k \
  --l0_target 176 \
  --out_dir clt_bundles/gemma-scope-2b-pt-res
```

README core-claims preset (layers `4/8/12/16/20/24`, local cache only):

```bash
python scripts/gemma_scope_to_clt.py \
  --preset readme_core_bundle \
  --width 16k \
  --revision fd571b47c1c64851e9b1989792367b9babb4af63 \
  --local_files_only \
  --out_dir clt_bundles/gemma-scope-2b-pt-res_sweep_smoke
```

## Reproducible runs (pinned artifacts + manifests)

`aom_eval.py` writes a JSON run manifest next to the CSV (`*.manifest.json`) to make runs replayable and auditable. The manifest records CSV provenance (`csv_sha256`, `csv_n_rows`) so you can detect post-hoc edits.

Pin Hugging Face artifacts (optional):

```bash
python aom_eval.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --revision <branch|tag|commit> \
  --tokenizer_revision <branch|tag|commit> \
  --csv_path results/qwen_pinned.csv
```

Offline / air-gapped use (no downloads):

```bash
python aom_eval.py --local_files_only --csv_path results/offline.csv
```

Security: `--trust_remote_code` is **off by default**. Only enable it if you trust the model repository, since it may execute custom code during loading. Run manifests redact `--system_prompt` and `--system_prompt_file` argv values by default.

Frozen protocol provenance (recommended for flagship studies):

```bash
python aom_eval.py \
  --model_name_or_path google/gemma-2-2b \
  --protocol_path configs/mom_flagship_protocol.yaml \
  --csv_path results/mom_flagship_gemma2.csv
```

When `--protocol_path` is set, runners compute and record `protocol_sha256` automatically in the CSV row and run manifest.
If `--protocol_sha256` is also provided, it must match the file hash (mismatch is a hard error).
For protocol-controlled knobs (for example `bootstrap_n`, `bootstrap_seed`, `ci`, `require_git`), runs fail closed if CLI values diverge from the frozen protocol.

## Token boundary hygiene (warn-only by default)

If prompts don’t end in whitespace/newline, tokenization at the prompt→continuation boundary can change when scoring separately (“token healing” risk). By default, `aom_eval.py` warns; you can make it strict with `--boundary_check error` or opt into deterministic normalization with `--normalize_boundaries`.

## Prompt modes (raw vs chat templates)

Default AoM datasets use full prompt strings:

- `--prompt_input full_prompt --prompt_mode raw` (default)

For chat models, you can interpret dataset prompt fields as user messages and render via the tokenizer’s chat template:

- `--prompt_input user_message --prompt_mode chat_template`
- Optional system prompt via `--system_prompt_file` (preferred) or `--system_prompt` (redacted in run manifests)

Run manifests store hashes only (e.g., `system_prompt_sha256`, `chat_template_sha256`) and never write raw prompt/system text to disk.

## Paper-mode runner (smoke / M1Max / cuda_validated)

This repo includes a convenience runner that generates a hardened “paper dataset” (with CF shams + COH controls) and runs reproducible evaluation presets:

```bash
# Offline end-to-end smoke check (builds a tiny local model; no downloads)
python scripts/run_paper.py smoke

# Long local run tuned for Apple Silicon (MPS)
python scripts/run_paper.py m1max

# Full run intended for validated CUDA GPU sweeps
python scripts/run_paper.py cuda_validated --attn_behavioral flash_attention_2
```

Optional CLT integration in paper-mode:
- `--run_clt_patching`: include CLT metrics in the primary `aom_eval.csv` stage.
- `--run_clt_stage`: run an extra DISAMB-only CLT stage writing `clt_cpt_disamb_only.csv`.

```bash
python scripts/run_paper.py smoke \
  --dry_run \
  --run_clt_stage \
  --clt_repo clt_bundles/gemma-scope-2b-pt-res \
  --clt_layers 12 \
  --clt_run_name average_l0_176
```

## CF / COH causal patching runners

Separate CLIs implement the causal interventions for AoM-CF and AoM-COH:

```bash
python aom_cf_patching.py --config configs/cf_patching_gpt2_paper.yaml
python aom_coh_patching.py --config configs/coh_patching_gpt2_paper.yaml
```

These write `results/*.csv` plus `results/*.manifest.json` with provenance (git commit, HF commit hash, versions, argv hash, dataset SHA-256s, wall time, seeds).

## MoM flagship TransformerLens flow

Export a reproducible flagship manifest for TL head-attribution/ablation analysis:

```bash
python scripts/export_mom_flagship_manifest.py \
  --model_name_or_path google/gemma-2-2b \
  --disamb_path data/disamb_pairs.jsonl \
  --selection_json results/mom_selection.json \
  --protocol_path configs/mom_flagship_protocol.yaml \
  --output_path results/mom_flagship_manifest.json
```

Run TL head attribution + mean-replacement ablations from that manifest:

```bash
python mom_circuits_tlens.py \
  --manifest_path results/mom_flagship_manifest.json \
  --protocol_path configs/mom_flagship_protocol.yaml \
  --results_path results/mom_flagship_tlens/circuits.json
```

Notes:
- `mom_circuits_tlens.py` requires `transformer-lens` installed.
- The exporter and TL runner both record protocol hash provenance (`protocol_sha256`) in their artifacts.
- Exporter and TL runner both fail closed on token-event instability by default (`prompt + label` must preserve the intended next-token event).

## Tables

Generate LaTeX tables from `results/` artifacts:

```bash
make tables
```

## Building the paper PDF

Requires [pandoc](https://pandoc.org/) and [tectonic](https://tectonic-typesetting.github.io/):

```bash
pandoc paper/MoM_paper.md -o paper/MoM_paper.pdf \
  --pdf-engine=tectonic \
  -V geometry:margin=1in \
  --from markdown+tex_math_single_backslash \
  --resource-path=paper
```

Generate figures from the repository root (requires matplotlib + a model checkpoint):

```bash
python scripts/generate_paper_figures.py --help
```
