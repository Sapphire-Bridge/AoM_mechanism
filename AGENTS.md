# AGENTS

## Purpose

- Read `CONTEXT.md` first for routing.
- Use `docs/context/` for the authoritative detailed context.
- Treat this file as working rules, not as the main architecture document.

## First Reads

- `CONTEXT.md`
- `docs/context/NAVIGATION.json`
- `docs/context/CONTEXT-003.md` for reproduction and runner work
- `docs/context/CONTEXT-002.md` for metrics, patching, CLT, or SAE logic
- `docs/context/CONTEXT-001.md` for manifests, seeds, datasets, and protocol provenance

## Safe First Commands

- `make check` for the fastest clean-clone validation path
- `make reviewer-check` for machine readiness and local asset checks
- `python scripts/check_evidence_contract.py`
- `python scripts/check_evidence_contract_fields.py`
- `python scripts/run_paper.py smoke`

## Working Rules

- Prefer the stable Make targets before reconstructing long manual command lines.
- Do not treat checked-in `results/` artifacts as scratch space unless the task explicitly requires regenerating them.
- Preserve provenance surfaces: dataset manifests, run manifests, protocol hashes, and evidence-contract links.
- If a result looks wrong, debug in this order: command path, provenance/manifests, runner, metric/intervention logic, then release/verification layer.
- Keep `CONTEXT.md` brief and route-focused. Put deeper explanations in `docs/context/`.

## High-Risk Areas

- `aom/repro.py`
- `aom/run_manifest.py`
- `aom/provenance/protocol.py`
- `aom_eval.py`
- `aom/metrics/disamb.py`
- `aom/metrics/clt_cpt.py`
- `scripts/run_paper.py`
- `scripts/verify_mom_paper.py`
- `scripts/check_evidence_contract.py`
