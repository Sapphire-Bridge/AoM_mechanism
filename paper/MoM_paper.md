# Mechanics of Meaning: Sparse Feature Interventions and the Basis Structure of Contextual Control in Transformers

## Abstract

*The Appearance of Meaning* (AoM) showed that contextualized token-in-context states causally control meaning-like preference margins in transformer models. Here we ask how that control is organized in representation space. In Gemma 2 2B, we compare matched raw residual and sparse autoencoder (SAE) interventions at shared `resid_post` sites under hard invariants and endpoint-native accounting. On lexical disambiguation (DISAMB), layer-4 SAE patching yields a larger mean donor-directed effect than raw patching and higher RMS-based effect-efficiency under scored-position disturbance accounting. The paired SAE-over-raw difference at layer 4 is directionally positive, but its 95% confidence interval includes zero. FP64 endpoint-native analysis attributes this paired difference to expected-set and other-set candidate log-probability terms; diagnostic \(\Delta\log Z\) is zero in the primary single-token regime. Matched-rank PCA recovers an intermediate effect, random orthogonal projections recover little, and an SAE reconstruction/residual split shows that the early result is not well explained by generic compression or by a simple monotonic reconstruction-fidelity account. This early, task-conditional pattern weakens or reverses at later layers and does not generalize uniformly to CF and COH, where raw residual patching is near parity or stronger on effect magnitude. We therefore conclude that AoM-relevant control is basis-sensitive, layer-dependent, and task-heterogeneous rather than uniformly sparse. The result extends AoM from causal localization to representational basis without implying meaning proper or sparse semantic atoms.

## §1 Introduction

Mechanistic interpretability asks where task-relevant information lives and how it is causally used. We focus on context-conditioned construction in an operational sense: controlled context interventions that alter next-token preferences while lexical form is held fixed. This treats meaning-like preference margins (AoM-relevant control) as a behavioral-causal target, not as a metaphysical claim.

### §1.1 From AoM to MoM

Previous work proceeded in two steps. First, transformer success was used to motivate a context-dependent, relational picture of linguistic competence. Second, *The Appearance of Meaning* (AoM) replaced that broad interpretive claim with a disciplined empirical program: AoM as an operational competence profile and Context-Primacy Thesis (CPT) as a causal claim about contextualized token-in-context states. AoM thus localized causal control of meaning-like preference margins without treating that result as evidence for meaning proper.

The remaining question is representational. If contextualized states causally control AoM-relevant behavior, in what internal basis is that control organized? One possibility is that the relevant signal is best treated as diffuse residual-stream activity. Another is that part of it is more selectively recoverable in a sparse feature basis. This paper addresses that question by comparing matched raw residual and SAE-basis interventions at shared `resid_post` sites.

This paper follows AoM (Borck, 2026). AoM established the operational explanandum—appearance of meaning as a controlled competence profile—and used sham-controlled activation patching across GPT-2 and Qwen2.5 to show that contextualized token-in-context states causally control meaning-like preference margins. The present paper fixes a single model/SAE pairing (Gemma 2 2B + Gemma Scope 16k) and asks in what representational basis that already-established control is most faithfully and efficiently recoverable.

### §1.2 Sparse-Precision Thesis (SPT)

**Sparse-Precision Thesis (SPT).** For some AoM-relevant behaviors, especially early lexical disambiguation, sparse-feature interventions can improve donor-directed effect per unit scored-position disturbance relative to raw residual-stream patching. SPT is explicitly layer-dependent and task-conditional. It does not predict a universal sparse advantage, nor does it imply context-free lexical atoms.

Throughout, “sparse feature basis” refers to a learned SAE feature coordinate system—an overcomplete dictionary with reconstruction error—rather than an exact linear change of basis.

We evaluate SPT by comparing matched raw residual and SAE-basis interventions at shared `resid_post` sites under hard invariants and endpoint-native accounting, then testing boundary conditions across additional AoM components.

Figures 1–3 summarize the main empirical arc: a six-layer raw-vs-SAE profile, a three-layer matched-control comparison including RECON/RESID stress tests, and the non-monotonic relation between reconstruction fidelity and behavioral recovery.

**Contributions**

1. **Representational extension of AoM/CPT.** We recast AoM's causal-localization result as a question about representational basis.
2. **Hard-gated substrate comparison.** We introduce a matched comparison of raw residual and SAE-basis interventions at shared `resid_post` sites under explicit activation, score, and control-equivalence invariants.
3. **Bounded early sparse-precision evidence.** On DISAMB, the strongest positive evidence appears at layer 4: the SAE arm has a larger mean effect and higher RMS-based effect-efficiency, while FP64 endpoint accounting and matched PCA/random/RECON/RESID controls constrain simpler alternatives.
4. **Boundary conditions.** The early, task-conditional advantage weakens or reverses later and does not generalize uniformly to CF and COH, yielding a layer-dependent, task-conditional rather than universal sparse account.

## §2 Related Work

### 2.1 Sparse autoencoders and learned feature dictionaries

This paper sits within the sparse autoencoder and dictionary-learning line of mechanistic interpretability. Bricken et al. (2023) argued that learned feature dictionaries can provide better units of analysis than individual neurons by decomposing activations into sparse features. Huben et al. (2024) showed that sparse autoencoders can recover interpretable features in language-model residual streams and can localize counterfactual behavior more finely than earlier decompositions. Templeton et al. (2024) scaled this program to Claude 3 Sonnet, emphasizing feature extraction at substantially larger scale. Relative to this literature, the present paper does not argue for monosemanticity or a universal sparse advantage. Its narrower question is comparative and causal: when raw residual and SAE-basis interventions are matched at the same site, when does the SAE substrate improve behavioral recovery or effect-efficiency, and when does it under-recover?

### 2.2 Causal mediation, causal tracing, and activation patching

Methodologically, the paper builds on the causal-intervention literature used to localize transformer computations. Vig et al. (2020) applied causal mediation analysis to transformer language models, identifying neurons and attention heads implicated in model behavior. Meng et al. (2022) used causal tracing in GPT to localize factual recall computations by corrupting inputs and restoring selected internal states. Conmy et al. (2023) systematized a mechanistic-interpretability workflow in which behavior, dataset, metric, and intervention granularity are chosen jointly, while Heimersheim and Nanda (2024) summarize best practices and interpretation hazards for activation patching. Makelov et al. (2023) add an important caution: subspace interventions can create an illusion of interpretability if behavioral effects arise through pathways that are not the intended mediator. Meloux et al. (2025) argue that mechanistic interpretability findings should be treated as statistical estimators with explicit variance and robustness reporting, a discipline adopted here through pair-clustered bootstrap uncertainty and cross-precision validation. The present paper adopts the matched-site, metric-first discipline of this literature, but shifts the question from localization alone to substrate comparison: raw residual patching versus SAE-basis writeback under shared invariants and endpoint accounting.

### 2.3 Positioning relative to AoM

The present work also differs in framing from purely engineering-focused interpretability papers. AoM treated meaning-like preference margins as an operational behavioral-causal target and argued that contextualized token-in-context states causally control those margins. MoM inherits that operational framing and asks a narrower representational question: in what internal coordinate system is this already-localized contextual control most faithfully or efficiently recoverable?

## §3 Methods

AoM established cross-family CPT regularities across GPT-2 and Qwen checkpoints. The present paper deliberately trades breadth for mechanistic resolution by fixing a single model/SAE pairing (Gemma 2 2B + Gemma Scope 16k) and a single shared intervention site (`resid_post`). The target here is not cross-model generality but basis-sensitive comparison under tightly matched intervention conditions.

### 3.1 Protocol constants

- **Model:** Gemma 2 2B (`google/gemma-2-2b`; run used snapshot `c5ebcd40...`).
- **SAE basis:** Gemma Scope 16k, same-site `resid_post` encode/decode/writeback.
- **Dataset:** DISAMB lexical disambiguation set (52 minimal pairs, 104 directions).
- **Patch site:** target token span at `resid_post`, with donor/receiver spans found by exact target-occurrence substring tokenization and required to have equal length with token-ID matching.
- **Scoring:** length-normalized logprob, strict finite checks, and pair-cluster bootstrap 95% CIs; replicate count is recorded per artifact family (locked CPU endpoint analyses use \(n=5000\); auxiliary analyses use their artifact-recorded \(n\)).

### 3.2 Behavioral score definition

For label \(\ell\) with candidate continuations \(\mathcal{C}_\ell\):

\[
S(\ell)=\log\left(\frac{1}{|\mathcal{C}_\ell|}\sum_{c\in \mathcal{C}_\ell}\exp\left(\frac{1}{|c|}\sum_{t\in c}\log p(t)\right)\right)
\]

and margin:

\[
m=S(\ell_{\text{exp}})-S(\ell_{\text{other}}), \qquad
\Delta m = m_{\text{patched}}-m_{\text{base}}.
\]

Intuitively, the metric asks whether a patch shifts normalized continuation probability mass from the other candidate set toward the expected candidate set at the same scored positions.

In this dataset, each label has exactly three candidates. Across all 312 candidates, tokenized continuation length is mostly single-token (308/312; mean \(1.013\), max \(2\)); no within-label candidate set shares a first token ID under the run tokenizer. All arms are scored on the same receiver prompt and continuation token positions; only the intervention forward path changes. On the primary DISAMB subset used in the endpoint-native analysis, where continuations are effectively single-token and expected/other candidate sets are disjoint at the scored position, this reduces to \(m=\log P_E-\log P_O\), since the \(-\log 3\) candidate-count constant cancels between labels. Here \(P_E\) and \(P_O\) denote expected-set and other-set candidate probability masses at the scored position on the primary single-token subset.

### 3.3 Comparability arms and hard gates

We run matched-site interventions:

- **Arm A (raw context-swap):** replace receiver activations with donor activations.
- **Arm B (raw context-delta):** add \(x_{\text{donor}}-x_{\text{receiver}}\).
- **Arm C (SAE \(\delta_1\)-decode):** latent patch, single-decode residual writeback.
- **Arm D (SAE safe-2decode):** error-preserving reconstruction variant.
- **Arm C$'$ (raw-equivalent control for C):** raw residual writeback using the SAE-decoded delta from Arm C.
- **Arm D$'$ (raw-equivalent control for D):** raw residual writeback using the SAE-decoded delta from Arm D.

Audit/control arms include raw-equivalent writeback checks and raw/SAE identity controls. Hard gates require:

- activation A$\approx$B, defined as \(\|x_A-x_B\|_2/\|x_A\|_2<10^{-6}\) at the patched site, where \(x_A\) is the raw-swap replacement and \(x_B\) is the raw-delta replacement,
- \(|\Delta m_A-\Delta m_B|<10^{-4}\),
- label-score tolerances \(<10^{-4}\) on \(|\Delta S_{\text{exp},A}-\Delta S_{\text{exp},B}|\) and \(|\Delta S_{\text{other},A}-\Delta S_{\text{other},B}|\),
- control-equivalence tolerances \(<10^{-4}\) for each SAE arm relative to its matched raw-equivalent control arm.

Additional control analyses, not part of the hard-gated A/B/C/D contract, are run on the same DISAMB comparability rows. These include (i) matched-rank PCA projection of the raw donor–receiver delta, (ii) matched-rank random orthogonal projections averaged over five seeds, and (iii) an SAE reconstruction/residual split implemented as reconstruction-only and residual-only writeback arms. These controls use the same patch site, scoring, and pair-cluster bootstrap protocol, but they are descriptive and are not included in invariant gating.

### 3.4 Analysis set and counts

From the locked CPU reruns summarized in Appendix A.2:

- \(n_{\text{rows,total}}=312\) (52 pairs × 2 directions × 3 layers),
- \(n_{\text{rows,all-arms-success}}=312\),
- \(n_{\text{rows,analysis-included}}=312\),
- \(n_{\text{invariant-fail}}=0\).

The endpoint-native primary-analysis set is \(288/312\) rows (48/52 pairs). The remaining 24 rows (4 pairs) are excluded by the primary applicability contract (single-token/disjoint-set constraints).

### 3.5 Confirmatory gate status

To prevent mechanism over-claiming, we separate directional diagnostics from the strict analysis tier:

- **Directional diagnostics (MPS fp16, relaxed invariants):** useful for workflow and figure drafting.
- **Strict mechanism claims (CPU strict):** required for manuscript claims about candidate-set log-probability-ratio accounting.

This gate is now satisfied in the strict CPU FP64 run summarized in Appendix A.2: all 288 primary-applicable rows pass cancellation/accounting checks (\(n_{\text{rows, primary cancellation fail}}=0\)). Relaxed MPS runs are retained for workflow only and are not used for main-text mechanism estimates.

### 3.6 Numerical precision validation (FP32 vs FP64)

We ran matched CPU reruns at FP32 and FP64 with identical data/model/configuration and compared both behavior and accounting.

| Metric | FP32 | FP64 |
|---|---:|---:|
| rows total | 312 | 312 |
| analysis included | 312 | 312 |
| primary-applicable rows | 288 | 288 |
| primary cancellation pass | 271 | 288 |
| primary cancellation fail | 17 | 0 |
| mean abs primary residual | \(1.399\times10^{-6}\) | \(1.040\times10^{-15}\) |
| max abs primary residual | \(7.016\times10^{-5}\) | \(5.329\times10^{-15}\) |

Behavioral layer estimates are effectively unchanged across precision; only accounting residual closure changes materially. Residual magnitudes are computed from the primary residual columns on primary-applicable rows. We therefore use FP64 for the main-text endpoint-native mechanism estimates and retain FP32 as a robustness check for behavioral effects.

### 3.7 Reproducibility note

Artifact identifiers, run manifests, and file-level provenance are reported in Appendix A.2 and the supplementary reproducibility materials. The main text reports only the experimental distinctions needed to interpret the claims.

### §3.8 Inferential status of analyses

To keep the evidential hierarchy explicit, the manuscript distinguishes four analysis tiers. The **primary inferential evidence** is the hard-gated DISAMB comparison at layers 4/8/12 with FP64 endpoint-native accounting. The **main interpretive support** consists of matched-rank PCA/random controls, RECON/RESID stress tests, and scored-position disturbance-efficiency analyses on the same DISAMB rows. The **boundary evidence** is the six-layer profile, the fixed-layer specificity analysis, and the CF/COH extensions, which address depth and task scope rather than re-establish the main mechanism claim. The Option C feature/head analyses in Appendix A are **exploratory** and are not used to establish the main substrate-comparison result.

## §4 Where basis-sensitive differences emerge

Section 4 provides descriptive layer-profile evidence about where raw and SAE interventions differ most strongly; the stricter substrate comparison follows in §5. In Gemma 2 2B, donor-directed effects are strongest in early/mid depth and attenuate later. Here, “Raw effect” and “SAE effect” denote mean margin shifts \(\Delta m=m_{\text{patched}}-m_{\text{base}}\) at each layer (raw-swap vs SAE patching, respectively). From matched six-layer runs on the DISAMB dataset (Appendix A.2):

| Layer | Raw effect (mean \(\Delta m\)) | SAE effect (mean \(\Delta m\)) | CRR (SAE/Raw) |
|---:|---:|---:|---:|
| 4  | 0.261 | 0.336 | 1.286 |
| 8  | 0.276 | 0.277 | 1.003 |
| 12 | 0.331 | 0.252 | 0.761 |
| 16 | 0.178 | 0.080 | 0.449 |
| 20 | 0.059 | 0.001 | 0.024 |
| 24 | 0.016 | 0.009 | n.m. |

![Six-layer DISAMB layer profile. Raw and SAE mean donor-directed margin effects are shown for layers 4, 8, 12, 16, 20, and 24. Error bars indicate pair-cluster bootstrap 95% CIs where available for the locked-comparability layers 4/8/12. The figure visualizes the early-to-late shift from modest SAE-over-raw recovery at layer 4 to clear SAE under-recovery at later depth.](../figures/fig1_disamb_layer_profile.pdf)

CRR is reported only when \(|\text{Raw effect}| \ge 0.05\); “n.m.” marks ratios that are not meaningful due to near-zero denominator. Layer 20 is near this threshold and should be interpreted cautiously.

For the overlapping locked-comparability layers (4/8/12), pair-clustered 95% CIs are available: raw \([0.045,0.477]\), \([0.071,0.477]\), \([0.127,0.520]\); SAE \([0.145,0.527]\), \([0.100,0.472]\), \([0.075,0.423]\).

This descriptive profile suggests an early window in which SAE recovery can match or exceed raw patching, followed by clear SAE under-recovery at later depth. Controls are clean in the same artifacts (raw sham \(\sim10^{-8}\text{–}10^{-7}\), SAE sham \(\sim10^{-7}\text{–}10^{-6}\), SAE identity-abs effect \(=0\) at all six layers). These runs are deterministic under this configuration (fixed data path, deterministic forward pass, bootstrap seed fixed at 42), so changing the run seed did not change aggregate SAE CPT outputs across seeds 0/1/42.

Global summaries are consistent with the layer table: the mean of the per-pair maximum layer effect is 0.7696 for raw and 0.7554 for SAE, and the mean layer of maximum effect shifts earlier under SAE (12.58 for raw vs 11.35 for SAE).

We report independent fixed-layer specificity results from an auxiliary fixed-layer run summarized in Appendix A.2, using a predetermined 25% depth rule (layer 6), not post-hoc best-layer selection:

- signed target effect: \(0.350\) \([0.134,\,0.565]\),
- signed control effect: \(0.099\) \([-0.004,\,0.200]\),
- signed delta: \(0.250\) \([0.021,\,0.462]\),
- signed win rate: \(0.587\) \([0.490,\,0.673]\).

Here, “signed control effect” is the off-target donor-to-receiver patch effect from deterministic matched-token/same-index fallback controls with target-buffer exclusions in the fixed-layer specificity protocol. This supports early-layer specificity under a fixed evaluation rule.

## §5 What supports the early sparse-precision regime on DISAMB

### §5.1 Motivation

When comparing raw residual interventions to SAE-basis interventions, a key summary is the context restoration ratio (CRR): SAE effect divided by raw effect on the same behavioral margin. In our comparability run, CRR exceeds 1 at layer 4, despite an SAE bottleneck. The core question is therefore not whether SAE preserves all signal; it is why a lossy basis can still improve the behavioral objective. This section presents the primary DISAMB analysis together with the main supporting control analyses.

### §5.2 Experimental setup

We analyze layers 4, 8, and 12 with Arms A–D under the hard-gated protocol above. The full intersection set is \(n=312\) rows; endpoint-native mechanism accounting is evaluated on the primary-applicable subset (\(n=288\) rows; 48/52 pairs).

### §5.3 Endpoint accounting

Token-level identity:

\[
\Delta\log p(t)=\Delta\text{logit}(t)-\Delta\log Z.
\]

For primary-applicable DISAMB rows (single-token, disjoint expected/other token sets at the same scored position), the endpoint-native margin is:
\[
m=\log P_E-\log P_O,\qquad
\Delta m=\Delta\log P_E-\Delta\log P_O.
\]
Here \(P_E\) and \(P_O\) denote expected-set and other-set candidate probability masses at the scored position.
For SAE-over-raw paired differences:
\[
d_{CA}=\Delta m_C-\Delta m_A.
\]
\[
d_{CA}=d_{CA,\log P_E}-d_{CA,\log P_O}+r,\qquad |r|\approx10^{-15}\ \text{(FP64)}
\]
with \(r\) denoting the measured accounting residual.

Interpretation target: the expected-set and other-set candidate log-probability terms are primary for mechanism claims on DISAMB. \(\Delta\log Z\) is retained as diagnostic telemetry and not used alone as endpoint-causal evidence.

To operationalize collateral distributional shift at scored positions, we use:
\[
D_{\mathrm{KL}}=\frac{1}{T}\sum_{t=1}^{T}D_{\mathrm{KL}}\!\left(p_{\text{base},t}\,\|\,p_{\text{patched},t}\right), \quad
D_{\mathrm{RMS}}=\frac{1}{T}\sum_{t=1}^{T}\sqrt{\frac{1}{|V|}\sum_{v\in V}\left(\Delta \text{logit}_{t,v}\right)^2}.
\]
We then define effect efficiency as \(\eta_D=\Delta m / D\) for a disturbance metric \(D\).

### §5.4 Results

#### 5.4.1 A$\approx$B invariant verification

A$\approx$B passed on all examples and layers; no invariant failures were observed (\(n=0\)). Both CPU reruns retained full analysis inclusion (\(312/312\) rows).

#### 5.4.2 Layer-level CRR and effects

| Layer | Effect\(_A\) (raw) | Effect\(_C\) (SAE) | CRR (C/A) |
|------:|--------------------:|-------------------:|----------:|
| 4     | 0.261               | 0.336              | 1.29      |
| 8     | 0.276               | 0.277              | 1.00      |
| 12    | 0.331               | 0.252              | 0.76      |

At layer 4, the mean SAE margin effect exceeds the mean raw effect; by layer 12, SAE under-recovers relative to raw.

#### 5.4.3 Cross-precision validation of effect vs accounting

| Metric | FP32 | FP64 |
|---|---:|---:|
| rows total | 312 | 312 |
| analysis included | 312 | 312 |
| primary-applicable rows | 288 | 288 |
| primary cancellation pass | 271 | 288 |
| primary cancellation fail | 17 | 0 |
| mean abs primary residual | \(1.399\times10^{-6}\) | \(1.040\times10^{-15}\) |
| max abs primary residual | \(7.016\times10^{-5}\) | \(5.329\times10^{-15}\) |

Behavioral estimates are stable across precision; accounting residual closure improves materially in FP64.

#### 5.4.4 Endpoint-native mechanism estimates (FP64 primary-applicable pairs)

| Layer | \(\Delta\Delta m\) mean [CI] | \(P(\Delta\Delta m>0)\) | \(d_{CA,\log P_E}\) | \(d_{CA,\log P_O}\) | \(d_{CA,\mathrm{diag}\log Z}\) |
|---:|---:|---:|---:|---:|---:|
| 4 | \(0.109087\) \([-0.006098,\,0.227374]\) | 0.9678 | 0.057962 | -0.051124 | 0.0 |
| 8 | \(0.003583\) \([-0.137334,\,0.144913]\) | 0.5188 | 0.065547 | 0.061964 | 0.0 |
| 12 | \(-0.071915\) \([-0.195146,\,0.039489]\) | 0.1104 | 0.035462 | 0.107377 | 0.0 |

Interpretation from this endpoint-native table: at L4, the paired SAE-over-raw difference is directionally positive but not conventionally decisive by a 95% CI criterion; L8 is near cancellation; L12 is directionally negative. We therefore treat the L4 result as directional evidence within the strict analysis set for a bounded early sparse-precision regime, not as definitive proof of SAE superiority.

#### 5.4.5 Matched compression controls and reconstruction/residual stress tests

To test whether the L4 difference is better explained by the SAE basis than by generic low-rank projection, we ran a locked control analysis on the full DISAMB comparability set (Appendix A.2). This run retains full inclusion (\(n_{\text{rows}}=312\), \(n_{\text{pairs}}=52\)), zero invariant failures, success on all required and optional arms, and pair-cluster bootstrap uncertainty with \(n=1000\). The added arms are matched-rank PCA, matched-rank random orthogonal projections (5 seeds), and an SAE reconstruction/residual split.

| Layer | Raw A | PCA | Random mean | RECON | RESID |
|---:|---:|---:|---:|---:|---:|
| 4 | \(0.2612\) \([0.0451,0.4770]\) | \(0.2127\) \([0.0611,0.3683]\) | \(0.0198\) \([0.0013,0.0373]\) | \(0.3359\) \([0.1449,0.5269]\) | \(-0.0552\) \([-0.1999,0.0913]\) |
| 8 | \(0.2762\) \([0.0705,0.4770]\) | \(0.1911\) \([0.0579,0.3250]\) | \(0.0066\) \([-0.0072,0.0201]\) | \(0.2770\) \([0.1003,0.4724]\) | \(0.0110\) \([-0.1220,0.1553]\) |
| 12 | \(0.3307\) \([0.1270,0.5197]\) | \(0.2190\) \([0.0967,0.3361]\) | \(0.0120\) \([-0.0065,0.0286]\) | \(0.2515\) \([0.0749,0.4227]\) | \(0.0795\) \([-0.0324,0.1908]\) |

![Three-layer matched-control comparison on DISAMB. For layers 4, 8, and 12, bars show Raw A, matched-rank PCA, matched-rank random orthogonal projection mean, SAE reconstruction-only writeback (RECON), and complementary residual-only writeback (RESID). The main qualitative pattern is RECON \(>\) A \(>\) PCA \(\gg\) Random with RESID mean-negative at L4, near-zero at L8, and mean-positive at L12.](../figures/fig2_controls_recon_resid.pdf)

At layer 4, the mean ordering is \(\text{RECON} > A > \text{PCA} \gg \text{Random}\); at layer 8, \(\text{RECON}\approx A > \text{PCA} \gg \text{Random}\); and at layer 12, \(A > \text{RECON} > \text{PCA} \gg \text{Random}\). Raw exceeds the random control at all three layers (\(d(\text{RAND}-A)=-0.2413\,[-0.4440,-0.0362],\ -0.2696\,[-0.4618,-0.0712],\ -0.3187\,[-0.4975,-0.1249]\)), while matched-rank PCA recovers an intermediate effect but its paired difference from raw still overlaps zero at each layer. Across five random seeds, the random-control mean effect remains near zero (seed-mean SD \(0.0153,\ 0.0117,\ 0.0082\) at layers 4/8/12). This pattern is not well captured by a simple random-subspace explanation and indicates that the early L4 difference is not well captured by generic matched-rank PCA compression.

The reconstruction/residual split sharpens the layer story. At L4, reconstruction-only writeback exceeds the raw mean effect and residual-only writeback is mean-negative; at L8, reconstruction-only is near raw and the residual is near zero; at L12, reconstruction-only under-recovers and the residual becomes mean-positive. Activation-level additivity holds to numerical precision (additivity error \(\sim1.7\times10^{-8}\) to \(\sim2.0\times10^{-8}\) across layers), so the reconstruction/residual split is exact in state space even though behavioral effects remain nonlinear downstream. Accordingly, RECON and RESID are best interpreted as perturbational stress tests rather than as an additive causal decomposition of behavioral effect. In this sense, the early SAE difference is consistent with retaining margin-helpful components while discarding margin-opposing ones, whereas later layers show the opposite failure mode.

#### 5.4.6 Layer-wise reconstruction fidelity and behavioral recovery

| Layer | relMSE | activation cosine | activation norm ratio | delta cosine |
|---:|---:|---:|---:|---:|
| 4 | 0.259 | 0.861 | 0.826 | 0.574 |
| 8 | 0.363 | 0.798 | 0.773 | 0.552 |
| 12 | 0.208 | 0.890 | 0.872 | 0.646 |

![Activation-reconstruction fidelity versus behavioral recovery across layers 4, 8, and 12. The x-axis plots activation cosine between the SAE reconstruction and the raw activation (higher indicates better fidelity), and the y-axis plots behavioral recovery relative to raw (CRR). The figure shows a non-monotonic relation: layer 12 has the highest activation cosine but the lowest CRR, whereas layer 8 has the lowest activation cosine yet near-parity recovery.](../figures/fig3_fidelity_vs_recovery.pdf)

Behavioral recovery does not monotonically track simple reconstruction fidelity across depth. Layer 12 has the best reconstruction fidelity by these metrics but the weakest SAE recovery relative to raw, whereas layer 8 has the worst fidelity yet near behavioral parity. This is inconsistent with a simple monotonic layerwise SAE-quality explanation for the 4/8/12 depth profile. The current evidence is more consistent with a representational transition: early projection can remove margin-opposing components, mid-depth projection is largely transparent, and later projection truncates useful components.

#### 5.4.7 Norm and directional diagnostics

Complementary L4 delta-space diagnostics (pair-clustered means, 95% CI) show:

- \(\|\Delta_{\text{sae}}\|/\|\Delta_{\text{raw}}\| = 0.701\) \([0.679,\,0.724]\),
- \(\cos(\Delta_{\text{sae}},\Delta_{\text{raw}})=0.574\) \([0.558,\,0.589]\),
- full-vocabulary RMS logit change at expected continuation positions: raw \(0.850\), SAE \(0.687\).

These delta-space measures are consistent with selective projection rather than scalar rescaling and are complementary to the layer-wise activation-fidelity statistics above.

#### 5.4.8 Disturbance and efficiency at layer 4

Layer-4 disturbance accounting (means from locked comparability rows):

| Quantity | Arm A (raw) | Arm C (SAE) | SAE/Raw |
|---|---:|---:|---:|
| \(\Delta m\) | 0.261 | 0.336 | 1.29 |
| \(D_{\mathrm{KL}}\) | 0.113 | 0.115 | 1.02 |
| \(D_{\mathrm{RMS}}\) | 0.850 | 0.687 | 0.81 |
| \(\eta_{\mathrm{KL}}=\Delta m / D_{\mathrm{KL}}\) | 2.32 | 2.91 | 1.26 |
| \(\eta_{\mathrm{RMS}}=\Delta m / D_{\mathrm{RMS}}\) | 0.307 | 0.489 | 1.59 |

Pair-cluster bootstrap confirms a positive RMS-efficiency advantage (SAE/raw \(=1.83\), 95% CI \([1.03,\,4.41]\)); KL-efficiency is directionally higher but less certain (SAE/raw \(=1.44\), 95% CI \([0.82,\,3.22]\)).

#### 5.4.9 Diagnostic \(\Delta\log Z\) channel (secondary)

On the primary-applicable endpoint-native split, \(d_{CA,\mathrm{diag}\log Z}=0\) at layers 4, 8, and 12 in FP64. This is expected in the dominant single-token regime and confirms that the current DISAMB margin mechanism does not depend on a nonzero diagnostic normalization term in this regime.

Earlier first-candidate \(\Delta\)logit/\(\Delta\log Z\) diagnostic telemetry is omitted from the main analysis because it is superseded by the endpoint-native FP64 decomposition used here.

#### 5.4.10 Layer dependence

Taken together, the control arms suggest a depth transition from beneficial projection at L4 (RECON \(>\) A and RESID mean-negative), through near-transparency at L8 (RECON \(\approx\) A and RESID \(\approx 0\)), to harmful truncation at L12 (RECON \(<\) A and RESID mean-positive). Because the PCA control is non-degenerate at L4/L8 but rank-saturated at L12, L12 PCA should be interpreted cautiously; this caveat does not affect the L4 result.

### §5.5 Interpretation

The most defensible interpretation is limited. On DISAMB at layer 4, SAE-basis patching directionally improves donor-directed recovery relative to raw patching and more clearly improves RMS-based effect-efficiency. The matched controls weaken two simple alternatives: generic low-rank compression, because matched-rank PCA is intermediate and random projections are near null; and a monotonic reconstruction-quality account, because layer 12 has the best reconstruction fidelity yet the weakest relative behavioral recovery. What remains open is whether the L4 directional pattern reflects better isolation of margin-helpful components or another structured projection effect not captured by the current baselines.

In endpoint-native terms, the L4 paired difference is associated with a relative increase in the expected-set log-probability term and a relative decrease in the other-set term, with diagnostic \(\Delta\log Z = 0\) in the primary single-token regime. At layer 8 these terms nearly cancel; at layer 12 the other-set term dominates and SAE under-recovers. The positive claim should therefore remain narrow: this setup reveals an early, task-conditional sparse-precision regime, not a universal sparse substrate for contextual control.

### §5.6 Limitations

1. **Task specificity:** decomposition shown on lexical disambiguation.
2. **Model specificity:** Gemma 2 2B + Gemma Scope 16k.
3. **Layer specificity:** early sparse-precision differences, later bottleneck losses.
4. **Metric specificity:** result is about normalized logprob margins, not raw-logit margins.
5. **Inferential strength at L4:** \(\Delta\Delta m\) is directionally positive with high \(P(\Delta\Delta m>0)\), but the 95% CI still includes zero, so the endpoint-native paired-difference claim remains directional rather than conventionally decisive. The observed between-pair variance suggests that a larger dataset would likely be required to resolve this paired difference at conventional significance.
6. **Control scope:** matched-rank PCA/random controls and the reconstruction/residual split materially strengthen the L4 interpretation; the sign-flipped residual at L4 is more structured than a simple readout-alignment account would predict, but these controls do not exhaust all possible non-SAE baselines or prove unique causal mediation.
7. **PCA caveat at L12:** the current PCA control is rank-saturated at L12 and should be treated as descriptive there.

## §6 Generalization and boundary conditions across AoM components

To assess scope rather than to duplicate the primary strict analysis, we ran matched CF/COH comparability experiments. These runs satisfy their prespecified task-specific comparability checks and retain full analysis inclusion (CF: \(60/60\) rows, 20 pairs; COH: \(480/480\) rows, 80 pairs), but they do not meet the stricter DISAMB FP64 endpoint-accounting standard. CF shows near-parity with mild SAE under-recovery at layers 4/8/12 (CRR \(=0.922\,[0.842,1.004],\,0.915\,[0.827,1.008],\,0.917\,[0.826,1.007]\)); COH shows more consistent under-recovery (CRR \(=0.870\,[0.767,0.966],\,0.869\,[0.784,0.970],\,0.855\,[0.766,0.946]\)). Because DISAMB and CF/COH come from different dataset construction pipelines, these analyses are best read as boundary evidence for limited generalization, not as a clean isolation of task structure. The appropriate cross-task conclusion is therefore modest: the sparse-precision pattern is most evident on DISAMB and does not generalize uniformly across AoM components.

Safety-null behavior in base models remains a boundary condition, motivating instruction-tuned follow-up rather than stronger base-model claims.

Supplementary cross-validation material (dataset-comparability notes and DISAMB robustness follow-ups) is reported in Appendix A.

## §7 Discussion

This paper's main empirical contribution is a matched substrate comparison under shared causal sites, invariants, and endpoint accounting. On DISAMB at layer 4, SAE-basis interventions directionally outperform raw residual patching on donor-directed effect and more clearly on RMS-based effect-efficiency, while matched-rank PCA and random projections recover less. This supports a bounded early sparse-precision regime. The claim should remain narrow: the paired L4 difference is directionally positive rather than conventionally decisive, and it is established in one model/SAE pairing and one primary task.

The controls narrow, but do not eliminate, alternative explanations. The L4 pattern is not well captured by random subspaces, generic matched-rank compression, or a monotonic reconstruction-fidelity account. At the same time, the present controls do not prove that SAE latents uniquely isolate the true mediator of the behavior; broader subspace-faithfulness and downstream nonlinearity concerns remain. RECON and RESID should therefore be read as perturbational stress tests, not as additive causal decompositions of behavioral effect.

Conceptually, the result concerns the basis structure of already-contextualized control. All interventions occur on token-in-context states after contextual integration, so positive SAE results do not imply context-free lexical carriers or sparse semantic atoms. The contribution is instead to show that the recoverability of AoM-relevant control depends on representational basis, layer, and task.

The methodological lesson is straightforward: substrate comparisons in mechanistic interpretability should be evaluated jointly by behavioral recovery, endpoint-native accounting, and disturbance accounting, with explicit matched controls and uncertainty reporting. Framed this way, MoM is best understood as a boundary-of-validity result: an early, task-conditional sparse-precision regime on DISAMB, near-parity at L8, under-recovery by L12, and limited transfer to CF/COH.

## §8 Conclusion

This paper extends AoM from causal localization to representational basis. In Gemma 2 2B, matched SAE-basis interventions on DISAMB reveal a bounded early sparse-precision regime: at layer 4 they directionally exceed raw residual patching on donor-directed effect and more clearly on RMS-based effect-efficiency, while later layers weaken or reverse the early pattern. Matched PCA/random controls and RECON/RESID stress tests suggest that the result is not well explained by generic compression or by simple reconstruction fidelity, but the paired L4 superiority claim remains directional rather than conventionally decisive. Across layers and tasks, AoM-relevant control appears basis-sensitive, layer-dependent, and task-heterogeneous rather than uniformly sparse. The appropriate conclusion is therefore modest: in this setup, some contextual control is more recoverable in an SAE feature substrate, especially for early disambiguation, but the mechanics of meaning are not exhausted by sparse features.

## Appendix A: Supplementary Reproducibility and Option C Results

This appendix summarizes supplementary Option C analyses and provides artifact pointers for reproducibility.

### A.1 Supplementary artifact inventory

| Component | Artifact(s) | Notes |
|---|---|---|
| Top-k recovery summary (base) | `results/mom_option_c_bigrun_fp16_google_gemma-2-2b/topk.summary.json`<br>`results/mom_option_c_bigrun_fp16_google_gemma-2-2b/topk.summary.manifest.json` | Manifest-validated big run. |
| Matched projection controls + reconstruction/residual stress test | `results/clt_raw_comparability_l4_l8_l12_controls_full.csv`<br>`results/clt_raw_comparability_l4_l8_l12_controls_full.summary.json` | Main-text control validation for §5. |
| Feature-family clustering (base) | `results/mom_option_c_bigrun_fp16_google_gemma-2-2b/feature_families_summary.csv`<br>`results/mom_option_c_bigrun_fp16_google_gemma-2-2b/feature_families_summary.manifest.json` | Exploratory; top-20 features only. |
| Feature selectivity (base) | `results/mom_option_c_bigrun_fp16_google_gemma-2-2b/feature_selectivity.csv` | Selectivity summary for the analyzed features. |
| Head attribution + ablation (base) | `results/mom_option_c_bigrun_fp16_google_gemma-2-2b/head_scores.csv`<br>`results/mom_option_c_bigrun_fp16_google_gemma-2-2b/head_ablation_rows.csv`<br>`results/mom_option_c_bigrun_fp16_google_gemma-2-2b/head_ablation_summary.csv`<br>`results/mom_option_c_bigrun_fp16_google_gemma-2-2b/head_qk_patterns.csv`<br>`results/mom_option_c_bigrun_fp16_google_gemma-2-2b/head_ablation_summary.manifest.json` | Head-level modulation evidence. |
| IT transfer check (partial) | `results/mom_option_c_bigrun_fp16_google_gemma-2-2b-it/topk.summary.json`<br>`results/mom_option_c_bigrun_fp16_google_gemma-2-2b-it/topk.summary.manifest.json`<br>`results/mom_option_c_bigrun_fp16_google_gemma-2-2b-it/head_scores.csv`<br>`results/mom_option_c_bigrun_fp16_google_gemma-2-2b-it/head_ablation_summary.csv`<br>`results/mom_option_c_bigrun_fp16_google_gemma-2-2b-it/head_ablation_summary.manifest.json` | Feature-family clustering not available in this IT run. |

Run metadata from manifests: model `google/gemma-2-2b`, device `mps`, dtype `float16`, split \(52 \to 26\) S / \(26\) E pairs, `random_control_mode=matched_bin`.

### A.2 Claim-to-artifact map

| Evidence ID(s) | Evidence unit | Artifact path(s) | Paper role | Notes |
|---|---|---|---|---|
| `R1`, `E1`, `E2`, `E25`, `E26`, `E27`, `E28`, `E31`, `E32` | Hard-gated comparability + decomposition | `results/mom_endpoint_plan/r1_full_cpu_f32.summary.json`; `results/mom_endpoint_plan/r1_full_cpu_f64.summary.json`; `results/mom_endpoint_plan/r1_full_cpu_f64.endpoint_decomp_summary_v2.json`; `results/mom_endpoint_plan/r1_full_cpu_f32.endpoint_decomp_summary_v2.json` | Main text | Core causal claim in §§3–5 (A$\approx$B gate, intersection set, endpoint-native candidate-set log-probability-ratio accounting, cross-precision stability, and primary-residual diagnostics). |
| `R7`, `E4`, `E13`, `E29`, `E33`, `E34`, `E35`, `E36` | Matched projection controls + reconstruction/residual stress test | `results/clt_raw_comparability_l4_l8_l12_controls_full.csv`; `results/clt_raw_comparability_l4_l8_l12_controls_full.summary.json`; `results/paper_lock_20260226T042208Z/clt_raw_comparability_l4_l8_l12.summary.json`; `results/paper_lock_20260226T042208Z/mom_quick_review_metrics.json` | Main text | §5 validation that the L4 difference is not well captured by matched-rank PCA/random projection; includes layer-wise fidelity, directional diagnostics, and disturbance-efficiency summaries. |
| `R2`, `R3`, `E5`, `E6` | Six-layer localization profile + controls | `results/overnight_mech_20260225T135850Z/gemma2b_raw_6layer_full_seed42.manifest.json`; `results/overnight_mech_20260225T135850Z/gemma2b_clt_6layer_full_seed42.manifest.json`; `results/overnight_mech_20260225T135850Z/gemma2b_clt_6layer_full_seed0.manifest.json`; `results/overnight_mech_20260225T135850Z/gemma2b_clt_6layer_full_seed1.manifest.json` | Main text | Layer-shape evidence in §4 (early stronger effects, late attenuation; sham/identity controls clean). Release bundle ships manifest-backed localization artifacts. |
| `R5`, `E7` | Fixed-layer specificity robustness run | `results/mom_overnight_gemma2b_sae_20260224T165425Z/gemma2b_sae.csv`; `results/overnight_mech_20260225T135850Z/gemma2b_raw_6layer_full_seed42.manifest.json` | Main text | Predetermined 25% depth rule (layer 6). Uses `data_paper_hardened_v2/disamb_pairs.jsonl`, so it is treated as robustness support rather than pooled with the six-layer table. |
| `E17`, `E18` | CF/COH task-axis boundary runs | `results/clt_raw_comparability_cf_l4_l8_l12_final_f32.summary.json`; `results/clt_raw_comparability_coh_l4_l8_l12_final_f32.summary.json` | Main text | Boundary evidence in §6; task-specific comparability passes, but these runs do not re-establish the strict DISAMB FP64 endpoint-accounting claim. |
| `E12` | Safety boundary condition (base/IT baselines) | `results/paper_lock_20260226T042208Z/gemma2b_clt_6layer_safety_seed42.csv`; `results/safety_it_20260226T074644Z/gemma2b_it_safety_baseline_seed42.csv`; `results/safety_it_naturalistic/gemma2b_it_safety_baseline_seed42.csv` | Main text | Boundary evidence in §6; avoid over-claiming safety-context commitment from the current base-model mechanism artifacts. |
| `R4`, `E8` | Top-k recovery concentration run | `results/mom_option_c_bigrun_fp16_google_gemma-2-2b/topk.summary.json`; `results/mom_option_c_bigrun_fp16_google_gemma-2-2b/topk.summary.manifest.json` | Appendix | Concentration evidence only (hundreds-to-thousands of features for recovery). |
| `R4`, `E9`, `E30` | Feature-family clustering and selectivity summary | `results/mom_option_c_bigrun_fp16_google_gemma-2-2b/feature_families_summary.csv`; `results/mom_option_c_bigrun_fp16_google_gemma-2-2b/feature_families_summary.manifest.json`; `results/mom_option_c_bigrun_fp16_google_gemma-2-2b/feature_selectivity.csv` | Appendix | Exploratory feature-family clustering and selectivity summaries for the analyzed feature set. |
| `R4`, `E10`, `E11` | Head attribution + ablation validation | `results/mom_option_c_bigrun_fp16_google_gemma-2-2b/head_scores.csv`; `results/mom_option_c_bigrun_fp16_google_gemma-2-2b/head_ablation_rows.csv`; `results/mom_option_c_bigrun_fp16_google_gemma-2-2b/head_ablation_summary.csv`; `results/mom_option_c_bigrun_fp16_google_gemma-2-2b/head_qk_patterns.csv`; `results/mom_option_c_bigrun_fp16_google_gemma-2-2b/head_ablation_summary.manifest.json` | Appendix | Head-level modulation in base model (sign is suppressive under this margin metric). |
| `R6`, `E14`, `E15`, `E16` | IT Option C transfer artifacts (partial) | `results/mom_option_c_bigrun_fp16_google_gemma-2-2b-it/topk.summary.json`; `results/mom_option_c_bigrun_fp16_google_gemma-2-2b-it/topk.summary.manifest.json`; `results/mom_option_c_bigrun_fp16_google_gemma-2-2b-it/head_scores.csv`; `results/mom_option_c_bigrun_fp16_google_gemma-2-2b-it/head_ablation_summary.csv`; `results/mom_option_c_bigrun_fp16_google_gemma-2-2b-it/head_ablation_summary.manifest.json` | Appendix | Preliminary transfer check; does not license strong causal-transfer claims. |
| `E19`, `E20`, `E21`, `E22`, `E23`, `E24` | DISAMB robustness follow-up analyses | `results/disamb_followups_20260227T123857Z/analysis/bootstrap_stability_l12_k20.csv`; `results/disamb_followups_20260227T123857Z/analysis/split_seed_stability_l12_k20.csv`; `results/disamb_followups_20260227T123857Z/analysis/split_seed_pairwise_jaccard_l12_k20.csv`; `results/disamb_followups_20260227T123857Z/analysis/subset_stability_l12_k20.csv`; `results/disamb_followups_20260227T123857Z/analysis/k_sweep_separation_l4_l8_l12.csv`; `results/disamb_followups_20260227T123857Z/analysis/scale_sweep_l12_k20.csv`; `results/disamb_followups_20260227T123857Z/analysis/feature_recurrence_l12_k20.csv`; `results/disamb_followups_20260227T123857Z/analysis/feature_recurrence_l12_k20_long.csv`; `results/disamb_followups_20260227T123857Z/analysis/report.md` | Appendix | Robustness axes used in Appendix A.8: bootstrap resampling, split seeds, context subsets, \(k\)-sweep, CLT-scale sweep, and feature recurrence. |

### A.3 Top-k concentration and recovery (base model)

| Layer | Gini | Recovery@k=50 | Recovery@k=500 | Recovery@k=1000 | Full effect |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.9617 | -0.1598 | 0.7454 | 0.9029 | 0.2453 |
| 8 | 0.9745 | 0.0213 | 0.7899 | 0.7720 | 0.1748 |
| 12 | 0.9551 | 0.5376 | 0.4797 | 0.7574 | 0.1805 |

Control comparison at \(k=500\):

| Layer | Top-k effect | Random-k (matched-bin) | Bottom-k |
|---:|---:|---:|---:|
| 4 | 0.1829 | 0.0382 | -0.0034 |
| 8 | 0.1380 | 0.0005 | -0.0007 |
| 12 | 0.0866 | 0.0164 | 0.0019 |

Interpretation: concentration is real but moderate. Effects are not captured by a handful of latents; recovery typically requires hundreds to low-thousands of features. At layer 4, \(k\approx 500\text{–}1000\) (about 3–6% of 16k) recovers most of full effect.

### A.4 Feature-family analysis summary (base model)

| Metric | Value |
|---|---:|
| Features analyzed | 20 |
| Mean \(|\)selectivity index\(|\) | 0.2820 |
| Max \(|\)selectivity index\(|\) | 0.5909 |
| Clusters found | 13 |
| Clustering stability (ARI mean) | 0.7518 |
| ARI 95% CI | [0.7287, 0.7735] |

The feature stage currently consumes top-ranked features exposed in the top-k summary (20 in this run), not the requested top-50 pool.

### A.5 Head attribution and ablation (base model)

Head-ablation validation was run at layer-4 feature attribution settings with `head_layers=1,2,3`, `top_h=5`, and E-split evaluation (\(n=26\) pairs).
Although the run was configured with `top_n_features=50`, the current top-k summary exposes 20 ranked features at layer 4, so the attribution stage used \(n_{\text{features}}=20\).

Reduction is defined as:
\[
\text{reduction}=\text{baseline context effect}-\text{ablated context effect}.
\]
So negative values mean ablation increased the context effect.

| Arm | Mean reduction | 95% CI | n |
|---|---:|---:|---:|
| Top-k heads | -0.3545 | [-0.6195, -0.1388] | 26 |
| Random-k heads | 0.0365 | [-0.1405, 0.2131] | 26 |
| Top-k minus Random-k | -0.3911 | [-0.6806, -0.1089] | 26 |

Top-ranked heads by selection score are concentrated in early layers (4/5 in layer 3): \((L3,H7)\), \((L2,H3)\), \((L3,H5)\), \((L3,H2)\), \((L3,H0)\). QK diagnostics show high target-to-context mass for three of these heads (mean mass: \(L3H0=0.972\), \(L3H7=0.932\), \(L3H5=0.924\)).

Interpretation: the ranked early heads are causally involved in context processing, but in this base-model margin metric their net role is suppressive (removing them tends to increase measured context effect). This supports a **modulation** claim, not a simple “these heads positively carry the effect” claim.

### A.6 Summary

These supplementary Option C analyses indicate feature-level non-uniformity, moderate top-k recoverability, and early-layer head-level causal modulation in the base model. Instruction-tuned Option C coverage is partial, so strong IT causal-transfer claims remain deferred.

### A.7 Preliminary Base-vs-IT Option C Comparison

We compared base and IT Option C artifacts at:

- base: `results/mom_option_c_bigrun_fp16_google_gemma-2-2b`,
- IT: `results/mom_option_c_bigrun_fp16_google_gemma-2-2b-it`.

Top-k / feature concentration:

- full context effect is larger in IT at all tested layers: \(L4: 0.414\) vs \(0.245\), \(L8: 0.510\) vs \(0.175\), \(L12: 0.624\) vs \(0.181\),
- top-\(k=50\) effect is larger in IT at \(L4/L8\) and similar at \(L12\): \(L4: 0.127\) vs \(-0.039\), \(L8: 0.085\) vs \(0.004\), \(L12: 0.094\) vs \(0.097\),
- concentration by mass-at-50 is slightly higher in IT at \(L4\) and lower at \(L8/L12\): \(L4: 0.288\) vs \(0.277\), \(L8: 0.282\) vs \(0.326\), \(L12: 0.155\) vs \(0.203\),
- top-20 feature overlap is moderate: \(11/20\) (L4), \(9/20\) (L8), \(12/20\) (L12).

Head attribution transfer:

- top selection heads are similar (\(4/5\) overlap): shared \((L2,H3),(L3,H2),(L3,H5),(L3,H7)\),
- selection-to-eval stability is high in both runs (Pearson \(r=0.867\) base, \(0.908\) IT),
- QK target-to-context mass remains concentrated in early heads, with IT strongest at \((L3,H7),(L3,H5),(L3,H6)\).

Ablation validation (causal head evidence):

- base: `topk_minus_randomk = -0.391` with 95% CI \([-0.681,-0.109]\),
- IT: `topk_minus_randomk = -0.340` with 95% CI \([-0.618,-0.042]\).

Both runs are directionally negative, so these ablations do not support a positive “top heads causally carry effect” claim under the current metric. The transfer evidence therefore supports stable head scoring and stronger IT feature-level effects, but not a positive causal-head-transfer claim.

### A.8 Cross-validation and robustness supplements

Comparability note: the fixed-layer specificity file referenced in §4 uses `data_paper_hardened_v2/disamb_pairs.jsonl`, while the six-layer raw-vs-SAE table in §4 uses `data/disamb_pairs.jsonl`. We therefore treat the fixed-layer specificity results as robustness support, not as a pooled estimate with the six-layer table.

DISAMB robustness follow-up runs (all saved under `results/disamb_followups_20260227T123857Z/`) strengthen this as a conditional causal claim rather than a single-run artifact. At \(L12,k=20\), pair-bootstrap resampling (\(n=4\)) gives moderate top-feature overlap with the base run (Jaccard mean \(0.484\), range \(0.379\) to \(0.538\)) and a small positive average top-\(k\) minus random-\(k\) margin (\(0.0157\)) with one negative replicate. Across independent split seeds \(\{0,1,2\}\), the same \(L12,k=20\) margin remains positive on average (\(0.0720\), variance \(4.36\times 10^{-4}\)) and pairwise top-20 overlap is moderate (Jaccard mean \(0.673\), range \(0.600\) to \(0.818\)). Context-type subsets show heterogeneity: clean, distractor, and paraphrase-distractor remain positive, while paraphrase-clean is slightly negative in this run (\(-0.0156\)). Hyperparameter sweeps preserve the qualitative depth pattern: \(L4\) margins are negative across tested \(k\), \(L8\) modestly positive, and \(L12\) strongest positive; \(L12,k=20\) also stays positive across `clt_scale` \(0.5/1.0/2.0\). Feature recurrence analysis indicates a stable nucleus (feature `13407` appears in 14/14 robustness runs), with a less stable tail.

## Reference

Borck, F. (2026). *The Appearance of Meaning (AoM): Context-Dependence and Semantic Competence in Transformer Architectures*.

Bricken, T., Templeton, A., Batson, J., et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Transformer Circuits Thread.

Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). *Towards Automated Circuit Discovery for Mechanistic Interpretability*. Advances in Neural Information Processing Systems 36.

Huben, R., Cunningham, H., Riggs Smith, L., Ewart, A., & Sharkey, L. (2024). *Sparse Autoencoders Find Highly Interpretable Features in Language Models*. International Conference on Learning Representations (ICLR 2024).

Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and Editing Factual Associations in GPT*. Advances in Neural Information Processing Systems 35.

Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Tamkin, A., Durmus, E., Hume, T., Mosconi, F., Freeman, C. D., Sumers, T. R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., & Henighan, T. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*. https://transformer-circuits.pub/2024/scaling-monosemanticity/

Anthropic. (2025a). *Tracing the thoughts of a large language model*. https://www.anthropic.com/research/tracing-thoughts-language-model

Anthropic. (2024). *The engineering challenges of scaling interpretability*. https://www.anthropic.com/research/engineering-challenges-interpretability

Heimersheim, S., & Nanda, N. (2024). *How to use and interpret activation patching*. arXiv:2404.15255. https://arxiv.org/abs/2404.15255

Makelov, A., Lange, G., & Nanda, N. (2023). *Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching*. arXiv:2311.17030. https://arxiv.org/abs/2311.17030

Meloux, M., Peyrard, M., & Portet, F. (2025). *Mechanistic Interpretability as Statistical Estimation: A Variance Analysis of EAP-IG*. arXiv:2510.00845. https://arxiv.org/abs/2510.00845

Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., & Shieber, S. (2020). *Investigating Gender Bias in Language Models Using Causal Mediation Analysis*. Advances in Neural Information Processing Systems 33.
