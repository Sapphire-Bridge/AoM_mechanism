# CONTEXT-PLAN

This repo is better partitioned by paper workflow than by generic package boundaries. The dominant question for a future agent is usually one of these:

- How does a run get seeded, bound to datasets/protocol hashes, and written out?
- How do raw, CLT, and SAE interventions actually compute effects?
- How does the paper runner assemble comparable artifacts and derived figures?
- How are claims checked against contracts and sanitized for release?
- Which extended analysis tools sit outside the core paper path but still depend on the same provenance rules?

## Partition Scheme

| Doc | Scope | Why this split exists |
| --- | --- | --- |
| `CONTEXT-001` | Core infra: config, seeds, manifests, datasets, protocol enforcement, model loading | These pieces are reused everywhere and explain why runs are traceable |
| `CONTEXT-002` | Experiment pipeline: behavioral eval, raw activation patching, CLT/SAE patching, patching protocols | This is the main technical core behind the paper’s mechanistic claims |
| `CONTEXT-003` | Paper orchestration and analysis: smoke/full runners, comparability, endpoint decomposition, figures | This is the shortest path from scripts to paper-facing outputs |
| `CONTEXT-004` | Publication contract and release verification | Paper support is governed by explicit contract and release tooling, not just by experiments |
| `CONTEXT-005` | Extended mechanistic tooling: top-k recovery, feature families, head attribution, TL circuits, scaling study | These are important, but not all are on the core reproduction path |

## Build Principles

- Keep claim-critical paths near the top of each doc.
- Record invariants that another agent should preserve before refactoring.
- Prefer file-level maps and concrete flows over abstract architecture prose.
- Call out artifact and cache assumptions explicitly because many scripts are offline or cache-sensitive.

## Handoff Heuristics

- For provenance bugs, read `CONTEXT-001` first, then the runner in `CONTEXT-003`.
- For numerical disagreements in patching or controls, read `CONTEXT-002`, then `CONTEXT-003`.
- For missing support artifacts, broken evidence IDs, or release questions, read `CONTEXT-004`.
- For questions about flagship circuits, CLT feature analyses, or model-family comparisons, read `CONTEXT-005`.

## Known Boundaries

- Some files participate in more than one story. For example, `scripts/run_paper.py` is orchestration, but it also encodes reproducibility assumptions. The doc split favors the question a future agent is most likely to ask, not strict ownership.
- `aom` is an importable package, but the repo is not organized like a normal library-first project. The package mostly exists to support research scripts with shared logic.

## Minimal Reading Paths

- Paper claim trace: `CONTEXT-004` -> `CONTEXT-003` -> `CONTEXT-002`
- Run/debug trace: `CONTEXT-003` -> `CONTEXT-001` -> `CONTEXT-002`
- Reproducibility trace: `CONTEXT-001` -> `README.md` -> `scripts/reviewer_quickcheck.py`
- Extended analysis trace: `CONTEXT-005` -> `CONTEXT-001` -> `CONTEXT-002`
