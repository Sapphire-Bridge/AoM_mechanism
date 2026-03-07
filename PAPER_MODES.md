# Paper Modes (Smoke / M1Max / A100)

This repo ships a single runner script that standardizes “paper-style” runs:

- `smoke`: offline end-to-end check (tiny local model + tiny datasets)
- `m1max`: long run intended for Apple Silicon (MPS)
- `a100`: full run intended for CUDA GPUs (e.g. A100)

## Canonical paper dataset (hardened)

All non-smoke modes generate (or reuse) a hardened dataset bundle with CF shams and COH ablation controls:

```bash
python scripts/run_paper.py m1max --data_dir data_paper_hardened_v1
```

This writes:

- `data_paper_hardened_v1/disamb_pairs.jsonl`
- `data_paper_hardened_v1/counterfactual.jsonl` (includes `expected_effect ∈ {shift,invariant}`)
- `data_paper_hardened_v1/coherence.jsonl` (includes matched `main` / `ablate_relevant` / `ablate_irrelevant`)
- `data_paper_hardened_v1/DATASET_MANIFEST.json` (SHA256 hashes + line counts)

## Running

```bash
# Offline end-to-end smoke check (no downloads)
python scripts/run_paper.py smoke

# M1Max preset (expects `--device mps` to work)
python scripts/run_paper.py m1max

# CUDA preset (flash attention optional for the behavioral run)
python scripts/run_paper.py a100 --attn_behavioral flash_attention_2
```

Outputs land in:

- `results/paper_smoke/`
- `results/paper_m1max/`
- `results/paper_a100/`

Each folder contains `RUN_MANIFEST.json` (command lines) and `results_report.md` (inventory summary).

## Notes / pitfalls for writeups

- CPT patching runs in these presets are **DISAMB-only** (`--cf_path "" --coh_path ""`). Do not interpret `aom_composite` in those CSVs.
- `--device_map` (multi-GPU sharding) is supported for behavioral runs, but patching/specificity assume a single-device model.
- If you use `--device_map`, install `accelerate` (Transformers requirement for `device_map`).

