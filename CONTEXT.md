# CONTEXT

## What This Repo Is

- Mechanistic-interpretability research repo for AoM / Mechanics of Meaning experiments and the companion MoM paper.
- Main output families are behavioral evaluation runs, raw activation patching, CLT/SAE intervention analyses, paper-support artifacts, and sanitized publication artifacts.
- Correctness depends on seed/determinism control, dataset and protocol provenance, and claim-to-artifact traceability.
- This file is only the front door. The authoritative context lives under `docs/context/`.

## Start Here

- If the task is paper reproduction, runner behavior, or result assembly, read `docs/context/CONTEXT-003.md`.
- If the task is patching, metrics, spans, CLT/SAE intervention logic, or suspicious effect sizes, read `docs/context/CONTEXT-002.md`.
- If the task is manifests, datasets, protocol hashes, seed control, or model provenance, read `docs/context/CONTEXT-001.md`.
- If the task is manuscript verification, evidence IDs, release packaging, or publication artifacts, read `docs/context/CONTEXT-004.md`.
- If the task is flagship analyses, head attribution, TransformerLens paths, or scaling studies, read `docs/context/CONTEXT-005.md`.
- If you need the broad repo map first, read `docs/context/SCAN-000.md`.

## Critical Entry Points

- `scripts/run_paper.py` — main smoke and staged paper-run orchestration path.
- `scripts/run_mom_paper.py` — reviewer-facing MoM reproduction wrapper.
- `scripts/reviewer_quickcheck.py` — fastest readiness gate for offline smoke, dry-run, and local asset checks.
- `aom_eval.py` — canonical behavioral evaluation entrypoint.
- `aom/repro.py` — seed and determinism handling.
- `aom/run_manifest.py` — run provenance schema and manifest writer.
- `aom/provenance/protocol.py` — protocol hash enforcement.
- `aom/metrics/disamb.py` — main raw DISAMB patching metric path.
- `aom/metrics/clt_cpt.py` — CLT patching, controls, and recovery logic.
- `scripts/verify_mom_paper.py` — support-artifact verification.
- `scripts/check_evidence_contract.py` — manuscript/evidence-contract consistency checks.
- `Makefile` — stable top-level command contract for reviewers and agents.

## Current Command Contract

- `make check` — fastest clean-clone validation path; runs tests, evidence checks, and `scripts/run_paper.py smoke`.
- `make reviewer-check` — best first command on a fresh reviewer machine; checks smoke path, paper dry-run, and local assets.
- `make reproduction` — reviewer-facing full MoM reproduction path using local files.
- `make paper-reproduction-gpu` — accelerated paper reproduction path when an accelerator is available.
- `make tables` — regenerate tables from result artifacts.
- `python scripts/run_paper.py smoke` — direct smoke entrypoint when bypassing Make.
- `python scripts/check_evidence_contract.py` and `python scripts/check_evidence_contract_fields.py` — direct manuscript/contract validation.
- `python scripts/verify_mom_paper.py` — direct support-artifact verification path.

## Debugging Order

1. Start with `make check` unless the task explicitly targets a deeper flow.
2. If the failure is environment or asset related, run `make reviewer-check` and inspect `scripts/reviewer_quickcheck.py`.
3. If the failure is in paper orchestration or artifact assembly, inspect `scripts/run_paper.py` and `docs/context/CONTEXT-003.md`.
4. If the failure is a metric or intervention discrepancy, inspect `aom_eval.py`, `aom/metrics/disamb.py`, `aom/metrics/clt_cpt.py`, and `docs/context/CONTEXT-002.md`.
5. If the failure smells like provenance drift, inspect `aom/repro.py`, `aom/run_manifest.py`, `aom/provenance/protocol.py`, and `docs/context/CONTEXT-001.md`.
6. If the failure is publication-facing, inspect `scripts/check_evidence_contract.py`, `scripts/verify_mom_paper.py`, and `docs/context/CONTEXT-004.md`.

## Detailed Context Files

- `docs/context/SCAN-000.md` — repo scan, anchor map, hotspots, entry order.
- `docs/context/CONTEXT-PLAN.md` — partition logic and reading heuristics.
- `docs/context/CONTEXT-001.md` — config, seeds, manifests, datasets, protocol enforcement.
- `docs/context/CONTEXT-002.md` — behavioral eval, raw patching, CLT patching, SAE patching.
- `docs/context/CONTEXT-003.md` — paper runners, comparability analysis, endpoint decomposition, figures, tables.
- `docs/context/CONTEXT-004.md` — paper, evidence contract, verification, sanitized release.
- `docs/context/CONTEXT-005.md` — extended mechanistic tooling, flagship path, scaling study.

## Machine Navigation

- `docs/context/NAVIGATION.json` is the machine-readable map for these docs.
