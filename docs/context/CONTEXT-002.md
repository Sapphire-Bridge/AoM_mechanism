# CONTEXT-002

## Scope

Files covered here:

- `aom_eval.py`
- `aom/interventions/activation_patching.py`
- `aom/interventions/clt_loader.py`
- `aom/interventions/clt_patch.py`
- `aom/interventions/sae_loader.py`
- `aom/interventions/sae_patching.py`
- `aom/interventions/patching/base.py`
- `aom/interventions/patching/disamb_protocol.py`
- `aom/interventions/patching/cf_protocol.py`
- `aom/interventions/patching/coh_protocol.py`
- `aom/interventions/patching/authority_protocol.py`
- `aom/interventions/patching/scoring.py`
- `aom/metrics/disamb.py`
- `aom/metrics/counterfactual.py`
- `aom/metrics/coherence.py`
- `aom/metrics/authority_game.py`
- `aom/metrics/clt_cpt.py`
- `aom/metrics/sae_patching.py`
- `aom/metrics/sae_decomposition.py`

## Architecture Overview

This is the core experiment layer. It turns rendered task examples into measurements, then optionally perturbs internal model states or learned latent features to estimate causal effect.

There are three main intervention families:

- Raw activation patching at decoder block outputs.
- CLT latent patching with learned encoder/decoder structure.
- SAE feature patching with sparse-feature interventions.

`aom_eval.py` is the canonical top-level evaluator. The lower modules supply the task-specific scorers and intervention kernels.

## Key Flows

### 1. Behavioral evaluation in `aom_eval.py`

- `aom_eval.py` prepares datasets, loads models, applies rendering and boundary policies, and writes row-level outputs plus a run manifest.
- `_prepare_datasets_for_eval()` wires task data into the evaluation loop.
- `run_eval_with_loaded()` is the main in-memory execution path once model/tokenizer state already exists.
- `run_once()` executes a concrete evaluation configuration.
- `main()` handles argument parsing, multi-model and multi-seed execution, protocol binding enforcement, and output writing.

Important behavior:

- Dataset loaders are manifest-aware.
- Protocol binding enforcement happens before the core run proceeds.
- Output rows retain extensive provenance so later comparisons can be joined back to exact inputs and runtime conditions.

### 2. Raw activation patching

- `aom/interventions/activation_patching.py` abstracts model-family differences and exposes the basic patching hooks.
- The architecture registry covers decoder families including Qwen, GPT-2, Gemma, Llama, Mistral, and GPT-NeoX style stacks.
- Key helpers include architecture detection, decoder-block lookup, layer counting, block-output extraction, and span-level patch application.
- `forward_with_patched_block_output_span()` and `prefill_with_patched_block_output_span()` are the key raw patching surfaces.

Important convention:

- Hidden-state indexing follows `hs[0] = embeddings` and `hs[layer + 1] = post-block(layer)`. A future refactor must preserve that mental model or update all callers.

### 3. Task-specific raw patching metrics

- `aom/metrics/disamb.py` is the most important patching metric file for the paper path.
- It computes continuation scores, margins, context-swap patching effects, and target-specificity controls.
- `compute_cpt_context_swap_patching()` is the main DISAMB raw patching path.
- `compute_cpt_target_specificity_control()` adds a critical control surface.
- Token spans are located from rendered text via substring-to-token-span mapping. If prompt templates change, span logic must be rechecked.

Related task metrics:

- `aom/metrics/counterfactual.py` and `aom/metrics/coherence.py` support the CF and COH tasks.
- `aom/metrics/authority_game.py` supports the authority task family.
- `aom/interventions/patching/scoring.py` factors common continuation-scoring helpers for patching protocols.

### 4. Protocolized patching

- `aom/interventions/patching/base.py` introduces the protocol abstraction.
- `PatchingCase`, `CaseSkip`, `ComparisonSpec`, and `ActivationPatchingProtocol` separate example construction from the generic patching runner.
- `run_activation_patching()` handles execution, sham controls, aggregation, and the common output surface.
- Task-specific protocols in `disamb_protocol.py`, `cf_protocol.py`, `coh_protocol.py`, and `authority_protocol.py` define donor/receiver/span construction rules.

This is important because it reduces the chance that each task silently reimplements its own intervention semantics.

### 5. CLT loading and CLT patching

- `aom/interventions/clt_loader.py` loads CLT bundles into `LinearCLT` plus normalized `CLTMetadata`.
- Loader behavior is resilient to partially missing config by inferring defaults from weights where possible.
- The default assumptions are same-site `resid_post` bundles and the `same_site_v1` patching mode, which matches the repo’s main comparability path.
- `aom/interventions/clt_patch.py` implements latent-space intervention policies such as identity replacement, full latent replacement, and dimension-restricted replacement.
- The CLT hook enforces hidden-dimension compatibility for same-site usage and supports reconstruction modes including error-preserving variants.

### 6. SAE loading and SAE patching

- `aom/interventions/sae_loader.py` loads GemmaScope-style SAEs and canonicalizes weight orientation when needed.
- `aom/interventions/sae_patching.py` applies span-level feature interventions and round-trip checks.
- Shape validation is strict around layer, site, and token-mask compatibility, which is necessary because SAE interventions are easy to misapply silently.

### 7. CLT-specific patching analysis

- `aom/metrics/clt_cpt.py` contains the CLT analog of context-swap patching plus recovery/control analysis.
- It includes deterministic pair splitting, sham and identity controls, top-k recovery utilities, and paired bootstrap logic.
- `compute_clt_cpt_context_swap_patching()` is the main paper-facing CLT patching entrypoint in this file.

## Research Invariants

- Rendering and boundary policy must remain aligned between baseline scoring and intervention scoring.
- Token spans must be computed on the exact rendered strings consumed by the model, not on source-text shortcuts.
- Architecture detection must agree with where hooks are attached. A wrong block mapping can produce plausible but invalid numbers.
- Raw and latent patching controls are part of the experiment definition, not optional diagnostics.
- Same-site `resid_post` assumptions for CLT comparability are structural. Mixing sites changes the interpretation of the effect sizes.
- Protocolized patching should remain the preferred pattern for new tasks. One-off patch loops weaken consistency.

## Interfaces

Main execution surfaces:

- `aom_eval.py`
- `run_activation_patching()`
- `compute_cpt_context_swap_patching()`
- `compute_cpt_target_specificity_control()`
- `compute_clt_cpt_context_swap_patching()`
- CLT and SAE forward/prefill patching helpers in `aom/interventions/*`

Key data structures:

- `PatchSite`, `PatchSpanSite`
- `PatchingCase`, `ComparisonSpec`, `ActivationPatchingProtocol`
- `LinearCLT`, `CLTMetadata`
- `GemmaScopeSAE`, `GemmaScopeSAEMetadata`

## Cached Artifacts And Provenance

- Raw patching depends on the exact base model and tokenizer loaded by the infrastructure layer.
- CLT patching depends on external CLT bundle files, which are large and stored separately from ordinary code.
- SAE patching depends on compatible SAE parameter files and metadata.
- All three paths should write outputs that can be joined back to dataset manifests, model provenance, and protocol hashes.

## Operational Notes / Gotchas

- `aom_eval.py` is long because it combines CLI handling, orchestration, and row emission. When debugging, isolate whether the bug is in preparation, scoring, or output writing before editing.
- Hidden-state layer numbering is a common source of off-by-one errors. Reconfirm the `hs[layer + 1]` convention before touching patch site logic.
- CLT and SAE loader code includes compatibility shims for bundle metadata. Do not remove them casually unless all checked-in bundles are migrated.
- Span patching correctness depends on tokenization boundaries. Prompt-template changes can invalidate intervention sites without raising hard errors.

## Claim Traceability

- Behavioral result generation: `aom_eval.py`
- Raw context-swap effect claims: `aom/interventions/activation_patching.py` plus `aom/metrics/disamb.py`
- Protocolized intervention claims across tasks: `aom/interventions/patching/base.py` plus task protocol modules
- CLT effect and recovery claims: `aom/interventions/clt_loader.py`, `aom/interventions/clt_patch.py`, `aom/metrics/clt_cpt.py`
- SAE intervention claims: `aom/interventions/sae_loader.py`, `aom/interventions/sae_patching.py`, `aom/metrics/sae_patching.py`

If a paper number looks wrong, this is the usual debugging order:

- Confirm the task rows and prompt rendering in `aom_eval.py`.
- Confirm the span and layer selection logic.
- Confirm the correct intervention family was invoked.
- Confirm sham and identity controls.
- Confirm downstream aggregation only after the above checks pass.
