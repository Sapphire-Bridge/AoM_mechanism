# Paper verification guide (AoM_mechanism)

> This guide documents the current verification surface of this public repository only.
> It is not intended to enumerate every historical or exploratory path that may have existed in earlier local runs.

This repo is the mechanics-focused artifact and verification surface for the Mechanics of Meaning (MoM) paper. This guide gives reviewers the current supported verification routes, the key artifact surfaces, and the main caveats needed to interpret the checked-in release artifacts correctly.

Primary sources in this repo:
- Manuscript: `paper/MoM_paper.md`
- Evidence contract: `MoM_evidence_contract.md`
- Repo routing: `README.md`
- Fast verification scripts: `scripts/check_evidence_contract.py`, `scripts/check_evidence_contract_fields.py`
- Full run verification: `scripts/verify_mom_paper.py`

## Fast verification path

```bash
make check
```

This is the fastest clean-clone validation path. It creates `.venv` if needed, installs `requirements.txt`, and runs:

- `python -m pytest -q`
- `python scripts/check_evidence_contract.py`
- `python scripts/check_evidence_contract_fields.py`
- `python scripts/run_paper.py smoke`

## Fresh reviewer machine

```bash
make reviewer-assets
make reviewer-check
```

Use this on a fresh connected machine when you want to validate local assets, the offline smoke route, and the paper-runner dry-run before attempting a longer reproduction.

## Canonical strict paper reproduction

```bash
make reproduction MOM_PAPER_ARGS="--run_root /tmp/mom_paper_review_run"
python scripts/verify_mom_paper.py --run_root /tmp/mom_paper_review_run
```

`make reproduction` is the canonical reviewer-facing MoM path. It runs `scripts/run_mom_paper.py --local_files_only`, writes a Markdown report plus JSON command log under the run root, and generates the support artifacts checked by `scripts/verify_mom_paper.py`.

`make mom-paper` remains as a legacy alias for `scripts/run_mom_paper.py`, but `make reproduction` is the preferred target because it forces `--local_files_only`.

## Optional accelerator sweep

```bash
make paper-reproduction-gpu PAPER_GPU_ARGS="--results_dir /tmp/paper_cuda_validated --local_files_only"
```

Use this to demonstrate accelerator-backed operability. It is not the canonical claim-verification path.

## Interpretation caveats

- `results/` is the curated sanitized public release surface for this repository, not the full internal historical results tree.
- The main-text six-layer raw-vs-SAE profile and the locked comparability artifacts use `data/disamb_pairs.jsonl`.
- The fixed-layer specificity robustness artifact uses `data_paper_hardened_v2/disamb_pairs.jsonl` and is interpreted separately rather than pooled with the six-layer profile.
- Use `MoM_evidence_contract.md` and Appendix A.2 for claim-to-artifact mapping before inferring semantics from filenames alone.

## Artifact surface to inspect

- `paper/MoM_paper.md`
- `MoM_evidence_contract.md`
- `results/`
- `figures/`
- `public_artifacts/RELEASE_MANIFEST.json`
- `public_artifacts/audit/`

Appendix A.2 in the manuscript maps claims to artifact paths. The evidence-contract scripts are the fastest check for paper/contract drift; the full reproduction route is the correct check for run-level support artifacts.
