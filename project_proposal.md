# When Models Lie to Please: Tracing Internal-External Alignment Failures in Language Models via Sparse Interpretability

## A Research Proposal Using Gemma Scope 2

**Author:** Victor Ashioya  
**Affiliation:** Bluedot Impact (AI Safety & Interpretability)  
**Date:** February 2026  
**Status:** Scoping / Pre-registration

---

## Abstract

Large language models routinely produce outputs that diverge from their internal representations — they "say" things they don't internally "believe." This divergence manifests in two related but mechanistically distinct failure modes: **unfaithful chain-of-thought reasoning** (where stated reasoning steps don't causally drive the final answer) and **sycophancy** (where internal correctness signals are suppressed to match user expectations). We hypothesize that these phenomena share a common upstream mechanism — a learned "override circuit" that suppresses internally computed answers in favor of contextually preferred outputs — but diverge in what triggers the override (implicit reasoning shortcuts vs. explicit social pressure).

Using **Gemma Scope 2**'s suite of sparse autoencoders (SAEs) and transcoders trained on instruction-tuned Gemma 3 models (1B–27B parameters), we propose to: (1) identify and characterize SAE features that activate during faithful vs. unfaithful reasoning, (2) trace the multi-layer circuits via transcoders that mediate sycophantic suppression of correct answers, (3) test whether CoT unfaithfulness and sycophancy share causal features or represent independent failure modes, and (4) develop feature-based classifiers and steering interventions that can detect and mitigate both failure types at inference time.

This work bridges two active but largely separate research threads — CoT faithfulness evaluation and sycophancy mitigation — by grounding both in mechanistic interpretability. It extends our prior work on CoT faithfulness detection (88% linear probe accuracy) and value-aligned confabulation evaluation to the circuit level.

---

## 1. Motivation & Research Gap

### 1.1 The Convergence Problem

Two bodies of literature have been developing largely in parallel:

**CoT Faithfulness.** Recent work shows that chain-of-thought reasoning is frequently unfaithful — models produce plausible reasoning that doesn't reflect their actual computational process. Turpin et al. (2023) demonstrated that biasing features in prompts systematically influence CoT without being mentioned. Arcuschin et al. (2025) showed this occurs even on realistic, unbiased prompts, with post-hoc rationalization rates of up to 13% in production models. Anthropic's system card evaluations found that reasoning models fail to mention clues they demonstrably used. METR's follow-up showed that complex reasoning requiring CoT is almost always faithful, but simple reasoning that can happen "in the model's head" is where unfaithfulness concentrates.

**Sycophancy.** Models systematically shift their outputs to match user-stated beliefs, even when internally representing the correct answer. Recent mechanistic work has found that sycophantic and genuine agreement are represented along *distinct axes* in hidden space, that sycophancy can be steered using activation vectors (Rimsky et al., 2024), and that it varies with input phrasing across layers. A key finding from the "Sycophancy Is Not One Thing" paper (2025) is that sycophantic agreement, genuine agreement, and sycophantic praise are *separable* in activation space.

**The missing link.** Both phenomena involve the model "knowing" one thing internally but "saying" another. Yet no work has examined whether they share mechanistic components. This matters for safety: if they share an override circuit, interventions on one might transfer to the other. If they're independent, we need separate detection systems.

### 1.2 Why Gemma Scope 2 Enables This Now

Previous work was limited to either (a) probing individual layers without cross-layer analysis, or (b) activation steering without understanding the underlying circuits. Gemma Scope 2 provides:

- **SAEs on every layer** of instruction-tuned Gemma 3 (1B, 4B, 12B, 27B), enabling layer-by-layer feature analysis of both phenomena
- **Transcoders (skip and cross-layer)** that track how features propagate across layers — critical for tracing multi-step override computations
- **Matryoshka training** producing more stable, interpretable features than prior SAE releases
- **Chat-specific tools** designed explicitly for analyzing refusal mechanisms and CoT faithfulness

### 1.3 Our Prior Work & Positioning

This project extends two prior lines:

1. **CoT Faithfulness Detection** — Our circuit discovery and linear probe work achieved 88% accuracy in detecting whether CoT reasoning faithfully reflects the model's internal process. This was done at the probe level; Gemma Scope 2 lets us go deeper to the circuit level.

2. **Value-Aligned Confabulation (VAC) Research** — Our framework for context-dependent evaluation of factually ungrounded outputs demonstrated that traditional metrics fail to distinguish harmful from beneficial confabulation. The current project provides the *mechanistic* explanation for why outputs diverge from internal states.

3. **Greater-Than Circuit SAEs** — Our SAE work on the greater-than circuit provides hands-on experience with the exact toolchain (JumpReLU SAEs, feature analysis, circuit discovery) that Gemma Scope 2 scales up.

---

## 2. Research Questions

### Primary

**RQ1:** Do CoT unfaithfulness and sycophancy activate overlapping SAE features in Gemma 3 instruction-tuned models, and if so, at which layers?

**RQ2:** Can we identify a causal "override circuit" — a multi-layer computational pathway traced via transcoders — that suppresses internally correct representations in favor of contextually preferred outputs?

### Secondary

**RQ3:** Does the override circuit (if it exists) operate differently when triggered by implicit reasoning biases (CoT unfaithfulness) vs. explicit social pressure (sycophancy)?

**RQ4:** How does the faithfulness-sycophancy relationship change across model scale (1B → 4B → 12B → 27B)?

**RQ5:** Can feature-based interventions (clamping, steering) developed for one failure mode transfer to mitigate the other?

---

## 3. Methodology

### 3.1 Phase 1: Feature Discovery (Weeks 1–3)

**Objective:** Identify SAE features associated with faithful/unfaithful reasoning and sycophantic/non-sycophantic responses.

#### 3.1.1 Dataset Construction

We construct three paired datasets, each containing matched prompts that elicit contrasting behaviors:

| Dataset | Faithful/Non-Sycophantic Version | Unfaithful/Sycophantic Version |
|---------|--------------------------------|-------------------------------|
| **CoT-Bias** | Standard math/logic questions | Same questions with answer-ordering bias (Turpin et al.) |
| **CoT-Contradiction** | "Is X bigger than Y?" | "Is Y bigger than X?" (same pair, testing for contradictory rationalizations per Arcuschin et al.) |
| **Sycophancy-Opinion** | Factual questions (no user opinion stated) | Same questions preceded by user's incorrect opinion |
| **Sycophancy-Pressure** | Model gives initial answer | User challenges correct answer ("Are you sure? I think it's actually Z") |
| **Cross-domain** | Mixed prompts spanning both failure types | For testing shared features |

Target: 500+ prompt pairs per dataset, balanced across difficulty and domain.

#### 3.1.2 Feature Activation Analysis

For each prompt pair:

1. Run forward pass through Gemma 3 IT (starting with 4B, scaling to 12B/27B)
2. Extract SAE feature activations at every layer using Gemma Scope 2 SAEs
3. Compute differential activation: `Δf = f_unfaithful - f_faithful` per feature per layer
4. Identify features with statistically significant differential activation (Bonferroni-corrected)
5. Cluster features by activation pattern across layers (early/mid/late emergence)

#### 3.1.3 Feature Characterization

For top-k differential features:
- Autointerpretation via Neuronpedia / manual inspection of max-activating examples
- Classify features as: *correctness signals*, *user-agreement signals*, *confidence modulators*, *reasoning step markers*, *override triggers*, *output formatters*
- Test monosemanticity via activation on held-out prompts

**Key hypothesis to test:** We expect to find features that function as "internal confidence" or "I know the correct answer" signals that are active in both faithful and non-sycophantic conditions, and suppressed (or overridden) in unfaithful and sycophantic conditions.

### 3.2 Phase 2: Circuit Tracing (Weeks 3–5)

**Objective:** Trace the multi-layer computational pathway from internal correctness representation to output override.

#### 3.2.1 Transcoder-Based Circuit Discovery

Using Gemma Scope 2's skip-transcoders and cross-layer transcoders:

1. For prompts where the model "knows" the correct answer (verified by probing mid-layer representations) but outputs something else:
   - Construct attribution graphs showing how correctness features connect to output features
   - Identify the layer(s) where the correctness signal is suppressed
   - Trace which features causally contribute to the suppression

2. Compare attribution graphs between:
   - CoT unfaithfulness cases (model rationalizes biased answer)
   - Sycophancy cases (model agrees with user's wrong answer)
   - Control cases (model correctly answers)

#### 3.2.2 Circuit Validation

For identified circuits:
- **Ablation studies:** Clamp candidate override features to zero; measure whether this restores faithful/non-sycophantic behavior
- **Activation patching:** Patch activations from faithful forward pass into unfaithful forward pass at specific layers; identify where the divergence occurs
- **Perturbation experiments:** Following Anthropic's "Biology of a Large Language Model" methodology — inhibit feature groups and measure effects on downstream features and model output

### 3.3 Phase 3: Shared Mechanism Testing (Weeks 5–7)

**Objective:** Determine whether CoT unfaithfulness and sycophancy share causal mechanisms.

#### 3.3.1 Cross-Condition Transfer

1. Identify features that are differentially active in CoT unfaithfulness
2. Clamp/steer those features during sycophancy prompts
3. Measure whether sycophancy is reduced (evidence of shared mechanism)
4. Repeat in reverse direction

#### 3.3.2 Representational Geometry

- Compute the "unfaithfulness direction" in activation space (mean difference between faithful/unfaithful activations)
- Compute the "sycophancy direction" similarly
- Measure cosine similarity between these directions at each layer
- Test whether they lie in the same subspace or orthogonal subspaces

#### 3.3.3 Scale Analysis (RQ4)

Repeat Phase 1 feature discovery across Gemma 3 1B, 4B, 12B, 27B:
- Do the same features emerge at all scales?
- Does the override circuit become more sophisticated at larger scales?
- At what scale do sycophancy and CoT unfaithfulness first become distinguishable?

### 3.4 Phase 4: Detection & Mitigation (Weeks 7–9)

**Objective:** Build practical tools that leverage mechanistic understanding.

#### 3.4.1 Feature-Based Classifiers

Train lightweight classifiers on SAE feature activations to detect:
- Whether current generation is faithful to internal reasoning
- Whether the model is being sycophantic vs. genuinely agreeing
- Whether an override circuit is currently active

Compare against baselines:
- Dense linear probes on raw activations (the GDM negative result baseline)
- Output-only classifiers (no access to internals)
- Existing CoT faithfulness metrics

#### 3.4.2 Inference-Time Interventions

Test three intervention strategies:
1. **Feature clamping:** Set override features to zero during generation
2. **Directional steering:** Add anti-sycophancy / pro-faithfulness vectors to residual stream
3. **Conditional activation steering (CAST):** Apply interventions only when classifier detects override activation

Evaluate on:
- Faithfulness improvement (does the model's CoT better reflect its reasoning?)
- Sycophancy reduction (does the model maintain correct answers under social pressure?)
- Capability preservation (MMLU, GSM8K to ensure no degradation)
- Over-correction risk (does the model become inappropriately disagreeable?)

---

## 4. Literature Map

### 4.1 Core References

#### CoT Faithfulness

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| Turpin et al., 2023 ("Language Models Don't Always Say What They Think") | CoT explanations influenced by biasing features models fail to mention; accuracy drops up to 36% | Foundational unfaithfulness demonstration; our bias dataset design |
| Arcuschin et al., 2025 ("CoT Reasoning In The Wild Is Not Always Faithful") | Unfaithful CoT on realistic prompts; post-hoc rationalization rates: GPT-4o-mini 13%, Haiku 3.5 7% | Extends to non-adversarial settings; contradiction dataset design |
| Anthropic, 2025 ("Reasoning Models Don't Always Say What They Think") | Reasoning models more faithful but not perfectly; RL may incentivize hiding reasoning | System-card level evaluation; motivates internal detection |
| METR, 2025 ("CoT May Be Highly Informative Despite Unfaithfulness") | Complex reasoning requiring CoT is almost always faithful; unfaithfulness concentrates in simple cases | Calibrates expectations; informs prompt difficulty design |
| Barez et al., 2025 ("Chain-of-Thought Is Not Explainability") | Framework for CoT faithfulness: procedural soundness, causal relevance, completeness | Evaluation framework we can adopt |
| Measuring CoT Faithfulness by Unlearning (EMNLP 2025) | Parameter-level faithfulness measurement via unlearning reasoning steps | Complementary method; our approach uses SAE features instead |

#### Sycophancy Mechanisms

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| "Sycophancy Is Not One Thing" (2025) | Sycophantic agreement, genuine agreement, and sycophantic praise are linearly separable in hidden space along distinct axes | Core mechanistic finding we extend to circuits |
| Rimsky et al., 2024 | Sycophancy can be steered using DiffMean activation vectors | Baseline steering method; we compare SAE-based steering |
| "Mitigating Sycophancy via Sparse Autoencoders" (OpenReview 2025) | SAF method on Gemma-2-2b-it: sycophancy reduced from 63% to 39%; layer 17 identified as key | Direct precedent; we scale to Gemma 3 with Gemma Scope 2 |
| Wang et al., 2025 (ICLR) | Latent representation analysis of sycophancy; disentangled from truthfulness | Representational geometry we extend |
| ELEPHANT (2025) | Social sycophancy beyond explicit statements; framing and moral sycophancy | Broader sycophancy taxonomy; dataset design |
| Consistency Training (Google, 2025) | BCT reduces sycophancy and jailbreak success by training consistency | Training-time intervention we contrast with inference-time |

#### Interpretability Infrastructure

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| Gemma Scope 2 Technical Report (2025) | SAEs + transcoders on all Gemma 3 layers, 270M–27B; Matryoshka training | Primary toolchain |
| Anthropic Circuit Tracing (2025) | Attribution graphs via cross-layer transcoders; applied to Claude 3.5 Haiku | Methodology template for our circuit analysis |
| "Biology of a Large Language Model" (2025) | Diverse circuit studies including hallucination, multi-step reasoning | Validation methodology (perturbation experiments) |
| GDM Negative Results for SAEs (2025) | SAE probes underperform dense probes for jailbreak detection | Important baseline; motivates circuit-level (not just probe-level) approach |
| Goodfire + Rakuten PII Detection (2025) | SAE probes outperform activation probes in production PII detection | Counterevidence to GDM negative result; context matters |
| Neuronpedia Circuits Landscape (Aug 2025) | Multi-org replication of circuit tracing; PLT vs CLT comparison | State of the art in circuit tools |

### 4.2 Secondary References

- Anthropic "Scaling Monosemanticity" (2024) — safety-relevant features including deception, sycophancy
- "On Passive-Scoping" (MIT Thesis, 2025) — SAE filters for safety scoping
- "Emergent Misalignment" (2025) — narrow fine-tuning causes broad misalignment detectable with SAEs
- Neel Nanda's 2025 field update — pragmatic vs. ambitious interpretability
- "Dissociation of Faithful and Unfaithful Reasoning" (Yee et al., 2024)
- Baker et al. (OpenAI, 2025) — Monitoring reasoning models for misbehavior

---

## 5. Project Structure

```
internal-external-alignment/
├── README.md
├── LICENSE
├── pyproject.toml
│
├── configs/
│   ├── models.yaml               # Gemma 3 model configs (1B, 4B, 12B, 27B)
│   ├── sae_configs.yaml           # Gemma Scope 2 SAE/transcoder selection
│   └── experiment_configs/
│       ├── phase1_features.yaml
│       ├── phase2_circuits.yaml
│       ├── phase3_transfer.yaml
│       └── phase4_interventions.yaml
│
├── data/
│   ├── raw/                       # Source datasets
│   ├── processed/                 # Paired prompt datasets
│   │   ├── cot_bias/
│   │   ├── cot_contradiction/
│   │   ├── sycophancy_opinion/
│   │   ├── sycophancy_pressure/
│   │   └── cross_domain/
│   └── activations/               # Cached SAE feature activations
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset_builder.py     # Paired dataset construction
│   │   ├── prompt_templates.py    # Bias injection, opinion prefixing
│   │   └── loaders.py
│   ├── features/
│   │   ├── extraction.py          # SAE feature extraction via Gemma Scope 2
│   │   ├── differential.py        # Differential activation analysis
│   │   ├── clustering.py          # Feature clustering across layers
│   │   └── characterization.py    # Autointerpretation pipeline
│   ├── circuits/
│   │   ├── attribution.py         # Attribution graph construction
│   │   ├── transcoder_tracing.py  # Cross-layer transcoder analysis
│   │   ├── validation.py          # Ablation & patching experiments
│   │   └── visualization.py       # Circuit diagram generation
│   ├── analysis/
│   │   ├── geometry.py            # Representational geometry (cosine similarity)
│   │   ├── transfer.py            # Cross-condition transfer experiments
│   │   ├── scale.py               # Cross-scale comparison
│   │   └── statistics.py          # Significance testing, effect sizes
│   ├── interventions/
│   │   ├── clamping.py            # Feature clamping
│   │   ├── steering.py            # Directional activation steering
│   │   ├── conditional.py         # CAST-style conditional interventions
│   │   └── evaluation.py          # Intervention evaluation suite
│   └── utils/
│       ├── gemma_scope.py         # Gemma Scope 2 loading utilities
│       ├── mishax_wrapper.py      # Mishax integration
│       └── neuronpedia.py         # Neuronpedia API integration
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_discovery.ipynb
│   ├── 03_circuit_tracing.ipynb
│   ├── 04_transfer_analysis.ipynb
│   ├── 05_interventions.ipynb
│   └── 06_scale_comparison.ipynb
│
├── experiments/
│   ├── results/                   # Experiment outputs
│   └── scripts/                   # Reproducible experiment runners
│
├── paper/
│   ├── main.tex
│   ├── figures/
│   └── references.bib
│
└── tests/
    ├── test_data.py
    ├── test_features.py
    └── test_circuits.py
```

---

## 6. Compute Requirements & Toolchain

### 6.1 Models & SAEs

| Component | Source | Size |
|-----------|--------|------|
| Gemma 3 4B-IT | HuggingFace | ~8GB |
| Gemma 3 12B-IT | HuggingFace | ~24GB |
| Gemma 3 27B-IT | HuggingFace | ~54GB |
| Gemma Scope 2 SAEs (4B) | `google/gemma-scope-2` | Variable per layer |
| Gemma Scope 2 Transcoders | `google/gemma-scope-2` | Variable |

### 6.2 Infrastructure

- **Primary compute:** Colab Pro+ or Kaggle for initial experiments (4B model); cloud GPU (A100/H100) for 12B/27B
- **Libraries:** `transformers`, `mishax` (GDM internal tool, open-sourced), `circuit-tracer` (Anthropic), `sae-lens` (optional)
- **Visualization:** Neuronpedia (interactive circuit exploration), custom matplotlib/plotly for paper figures

### 6.3 Estimated Timeline

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| Dataset construction | Week 1 | 2,500+ paired prompts across 5 datasets |
| Feature discovery (4B) | Weeks 1–3 | Catalog of differential features + layer profiles |
| Circuit tracing (4B) | Weeks 3–5 | Attribution graphs for override pathways |
| Transfer experiments | Weeks 5–7 | Shared mechanism evidence (or refutation) |
| Scale analysis | Week 6 (parallel) | Cross-scale feature comparison |
| Interventions & eval | Weeks 7–9 | Classifier + steering results |
| Paper writing | Weeks 8–10 | Preprint draft |

---

## 7. Expected Contributions

1. **Mechanistic taxonomy:** First systematic characterization of SAE features involved in both CoT unfaithfulness and sycophancy within a single model family
2. **Circuit-level explanation:** Attribution graphs showing *how* models suppress correct answers — going beyond probe-level detection
3. **Shared mechanism test:** Empirical answer to whether unfaithfulness and sycophancy share computational substrates
4. **Practical tools:** Feature-based classifiers and steering methods benchmarked against existing baselines
5. **Scale analysis:** How internal-external alignment failures evolve from 1B to 27B parameters

---

## 8. Risk Mitigation & Contingency

| Risk | Probability | Mitigation |
|------|-------------|------------|
| SAE features too polysemantic to interpret | Medium | Use Matryoshka SAEs at multiple widths; fall back to attention-head analysis |
| No shared mechanism between CoT and sycophancy | Medium | This is itself a valuable finding; publish as negative result with clear implications |
| Gemma 3 4B too small for sycophancy to manifest | Low | DeepMind neg results blog showed effects at small scale; scale up to 12B early if needed |
| Compute insufficient for 27B analysis | Medium | Focus on 4B/12B; use 27B only for targeted validation of key findings |
| Transcoders produce uninterpretable graphs | Medium | Fall back to per-layer SAE analysis + activation patching (standard mech interp toolkit) |
| GDM's own negative SAE result applies to our setting | Medium | Our approach differs: circuit-level analysis, not just probing; PII detection shows context matters |

---

## 10. Open Questions for Brainstorming

- Should we include **refusal** as a third failure mode? (Refusal = model "knows" it can answer but outputs refusal; structurally similar to sycophancy in reverse)
- How do we handle the GDM team's pivot away from SAEs? Our approach is fundamentally different (circuit-level, not downstream task probing), but we need to address this clearly
- Should we use Anthropic's `circuit-tracer` library alongside Gemma Scope 2, or build our own tooling?
- Could we connect this to the **emergent misalignment** literature — does narrow misalignment training activate the same override circuits?
- For the Indaba connection: does the override circuit behave differently for multilingual prompts? (African language sycophancy vs. English sycophancy)

---

*This document is a living research scope. Last updated: February 2026.*
