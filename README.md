# When Models Lie to Please: Tracing Internal-External Alignment Failures via Sparse Interpretability

**Author:** Victor Ashioya  
**Affiliation:** Bluedot Impact (AI Safety & Interpretability)  
**Status:** Pre-registration / Active Research  
**Last Updated:** April 2026

---

## Overview

Language models routinely produce outputs that diverge from their internal representations. This divergence manifests in two mechanistically related failure modes:

- **Unfaithful chain-of-thought (CoT):** Stated reasoning steps do not causally drive the final answer. The model rationalizes post-hoc rather than reasoning genuinely.
- **Sycophancy:** Internally represented correct answers are suppressed to match user-stated beliefs or social pressure.

The central hypothesis of this project is that both phenomena share a common upstream mechanism — a learned **override circuit** that suppresses internally computed answers in favor of contextually preferred outputs — but are triggered by different inputs (implicit reasoning shortcuts vs. explicit social pressure).

Using **Gemma Scope 2**'s suite of sparse autoencoders (SAEs) and transcoders trained on instruction-tuned Gemma 3 models (1B–27B parameters), this project:

1. Identifies and characterizes SAE features that activate during faithful vs. unfaithful reasoning
2. Traces multi-layer circuits via transcoders that mediate sycophantic suppression of correct answers
3. Tests whether CoT unfaithfulness and sycophancy share causal features or represent independent failure modes
4. Develops feature-based classifiers and steering interventions for detection and mitigation at inference time

This work bridges two active but largely separate research threads — CoT faithfulness evaluation and sycophancy mitigation — by grounding both in mechanistic interpretability.

---

## Why This Matters

The internal-external alignment problem is a core safety concern. A model that reliably says what it internally represents is interpretable; one that systematically diverges is not. The existing literature treats CoT faithfulness and sycophancy as separate problems with separate toolkits. That framing misses the possibility that a single circuit-level mechanism drives both. If it does, interventions on one may transfer to the other, and both phenomena can be monitored with shared instrumentation.

The key methodological advance over prior work is moving from **probe-level detection** (linear probes on residual stream activations) to **circuit-level attribution** (cross-layer transcoder graphs showing the actual computational pathway). This directly addresses the GDM team's finding that SAE probes underperform dense probes for downstream task detection: we are not using SAEs as classifiers, we are using them as microscopes.

---

## Research Questions

**Primary**

- **RQ1:** Do CoT unfaithfulness and sycophancy activate overlapping SAE features in Gemma 3 instruction-tuned models, and if so, at which layers?
- **RQ2:** Can we identify a causal "override circuit" — a multi-layer computational pathway traced via transcoders — that suppresses internally correct representations in favor of contextually preferred outputs?

**Secondary**

- **RQ3:** Does the override circuit operate differently when triggered by implicit reasoning biases (CoT unfaithfulness) vs. explicit social pressure (sycophancy)?
- **RQ4:** How does the faithfulness-sycophancy relationship change across model scale (1B → 4B → 12B → 27B)?
- **RQ5:** Can feature-based interventions developed for one failure mode transfer to mitigate the other?

---

## Repository Structure

```
when_models_lie_to_please/
├── configs/
│   ├── models.yaml                    # Gemma 3 model configs (1B, 4B, 12B, 27B)
│   ├── sae_configs.yaml               # Gemma Scope 2 SAE/transcoder selection
│   └── experiment_configs/
│       ├── phase1_features.yaml       # Feature discovery configuration
│       ├── phase2_circuits.yaml       # Circuit tracing configuration
│       ├── phase3_transfer.yaml       # Cross-condition transfer configuration
│       └── phase4_interventions.yaml  # Intervention evaluation configuration
│
├── data/
│   ├── raw/                           # Source datasets (not committed)
│   ├── processed/                     # Paired prompt datasets
│   │   ├── cot_bias/                  # Turpin-style biased prompts
│   │   ├── cot_contradiction/         # Arcuschin-style contradiction prompts
│   │   ├── sycophancy_opinion/        # User-stated incorrect opinion prompts
│   │   ├── sycophancy_pressure/       # Post-answer social pressure prompts
│   │   └── cross_domain/             # Mixed prompts for shared feature testing
│   └── activations/                   # Cached SAE feature activations (not committed)
│
├── src/
│   ├── data/
│   │   ├── dataset_builder.py         # Paired dataset construction
│   │   ├── prompt_templates.py        # Bias injection, opinion prefixing
│   │   └── loaders.py                 # Dataset loading utilities
│   ├── features/
│   │   ├── extraction.py              # SAE feature extraction via Gemma Scope 2
│   │   ├── differential.py            # Differential activation analysis
│   │   ├── clustering.py              # Feature clustering across layers
│   │   └── characterization.py        # Autointerpretation pipeline
│   ├── circuits/
│   │   ├── attribution.py             # Attribution graph construction
│   │   ├── transcoder_tracing.py      # Cross-layer transcoder analysis
│   │   ├── validation.py              # Ablation & activation patching experiments
│   │   └── visualization.py           # Circuit diagram generation
│   ├── analysis/
│   │   ├── geometry.py                # Representational geometry (cosine similarity)
│   │   ├── transfer.py                # Cross-condition transfer experiments
│   │   ├── scale.py                   # Cross-scale comparison (1B → 27B)
│   │   └── statistics.py              # Significance testing, effect sizes
│   ├── interventions/
│   │   ├── clamping.py                # Feature clamping
│   │   ├── steering.py                # Directional activation steering
│   │   ├── conditional.py             # CAST-style conditional interventions
│   │   └── evaluation.py              # Intervention evaluation suite
│   └── utils/
│       ├── gemma_scope.py             # Gemma Scope 2 loading utilities
│       ├── mishax_wrapper.py          # Mishax integration
│       └── neuronpedia.py             # Neuronpedia API integration
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
│   ├── results/                       # Experiment outputs (not committed)
│   └── scripts/                       # Reproducible experiment runners
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

## Methodology

### Phase 1: Feature Discovery (Weeks 1–3)

Five paired datasets are constructed, each containing matched prompts eliciting contrasting behaviors:

| Dataset | Control Version | Treatment Version |
|---------|----------------|-------------------|
| `cot_bias` | Standard math/logic | Same with answer-ordering bias (Turpin et al.) |
| `cot_contradiction` | "Is X > Y?" | "Is Y > X?" (same pair) |
| `sycophancy_opinion` | Factual question, no opinion | Same preceded by user's incorrect opinion |
| `sycophancy_pressure` | Model gives initial answer | User challenges correct answer |
| `cross_domain` | Mixed, for shared feature detection | Mixed, treatment version |

For each prompt pair, we extract SAE feature activations at every layer via Gemma Scope 2, compute differential activation `Δf = f_treatment - f_control`, and identify features with statistically significant differential activation (Bonferroni-corrected). Features are clustered by activation pattern across layers and characterized via Neuronpedia autointerpretation.

**Central hypothesis:** Features functioning as "internal confidence" or "I know the correct answer" signals will be active in control conditions and suppressed or overridden in treatment conditions.

### Phase 2: Circuit Tracing (Weeks 3–5)

Using Gemma Scope 2's skip-transcoders and cross-layer transcoders, we construct attribution graphs for prompts where the model demonstrates internal knowledge of the correct answer (verified by probing mid-layer representations) but outputs an incorrect or sycophantic response.

Circuit validation proceeds via:
- **Ablation studies:** Clamp candidate override features to zero; measure behavioral restoration
- **Activation patching:** Patch activations from faithful forward pass into unfaithful forward pass at identified layers
- **Perturbation experiments:** Following the methodology in "Biology of a Large Language Model" — inhibit feature groups and measure downstream effects

### Phase 3: Shared Mechanism Testing (Weeks 5–7)

Cross-condition transfer: features identified as differentially active in CoT unfaithfulness are clamped during sycophancy prompts, and vice versa. We also compute the geometry of failure modes — the "unfaithfulness direction" and "sycophancy direction" in activation space — and measure their cosine similarity at each layer to determine whether they lie in the same subspace or orthogonal subspaces.

Scale analysis repeats Phase 1 across Gemma 3 1B, 4B, 12B, 27B to characterize how the override circuit (if it exists) evolves with scale.

### Phase 4: Detection and Mitigation (Weeks 7–9)

Lightweight classifiers are trained on SAE feature activations to detect faithful/sycophantic generation states. Three inference-time intervention strategies are tested: feature clamping, directional activation steering (extending Rimsky et al. 2024), and conditional activation steering (CAST). All interventions are evaluated on faithfulness improvement, sycophancy reduction, capability preservation (MMLU, GSM8K), and over-correction risk.

---

## Dataset Details

All datasets target 500+ prompt pairs, balanced across difficulty and domain. The construction pipeline is in `src/data/dataset_builder.py`.

Source material for each dataset:
- `cot_bias`: Derived from GSM8K, MMLU, and logical reasoning benchmarks with systematic bias injection following Turpin et al. (2023)
- `cot_contradiction`: Paired numerical comparison and factual questions; both orderings generated and verified to have a single correct answer
- `sycophancy_opinion`: TruthfulQA and factual QA datasets with incorrect user-opinion prefixes injected per Anthropic (2023) methodology
- `sycophancy_pressure`: Two-turn dialogues where the second turn challenges the model's correct first-turn answer
- `cross_domain`: Stratified sample across all four categories above

---

## Toolchain

| Component | Source |
|-----------|--------|
| Gemma 3 IT (1B, 4B, 12B, 27B) | `google/gemma-3-{1b,4b,12b,27b}-it` on HuggingFace |
| Gemma Scope 2 SAEs + transcoders | `google/gemma-scope-2` on HuggingFace |
| Mishax (GDM activation patching) | Open-sourced; used for activation patching experiments |
| circuit-tracer (Anthropic) | Attribution graph construction |
| Neuronpedia | Interactive feature inspection; autointerpretation API |
| sae-lens | Optional; SAE loading and analysis utilities |

Primary compute: Colab Pro+ / Kaggle for 4B experiments; A100/H100 for 12B and 27B. Activation caches for large models are stored externally and not committed to this repository.

---

## Installation

```bash
git clone https://github.com/[repo]/when_models_lie_to_please
cd when_models_lie_to_please
pip install -e ".[dev]"
```

Requires Python 3.11+. GPU with at least 24 GB VRAM for the 4B model with SAEs loaded. See `configs/models.yaml` for per-model memory requirements.

Environment variables expected:

```
HUGGINGFACE_TOKEN=...          # For gated Gemma 3 access
NEURONPEDIA_API_KEY=...        # For autointerpretation pipeline
WANDB_API_KEY=...              # For experiment tracking (optional)
```

---

## Quickstart: Full Pipeline

The five scripts must be run in order. Each phase produces a timestamped run directory under `experiments/results/` and the subsequent phase takes that run ID as input.

```bash
# Step 0: Build paired prompt datasets (run once)
python experiments/scripts/build_datasets.py \
    --output-dir data/processed \
    --min-pairs 500

# Step 1: Feature discovery — identify differential SAE features per layer
python experiments/scripts/run_phase1.py \
    --config configs/experiment_configs/phase1_features.yaml
# Outputs to experiments/results/phase1/<run_id>/
# Note the run_id printed at end of run — needed for Steps 2–4.

# Step 2: Circuit tracing — attribution graphs via cross-layer transcoders
python experiments/scripts/run_phase2.py \
    --config configs/experiment_configs/phase2_circuits.yaml \
    --phase1-run <run_id_from_step1>
# Outputs to experiments/results/phase2/<run_id>/

# Step 3: Shared mechanism testing — transfer experiments + representational geometry
python experiments/scripts/run_phase3.py \
    --config configs/experiment_configs/phase3_transfer.yaml \
    --phase1-run <run_id_from_step1>
# Outputs to experiments/results/phase3/<run_id>/

# Step 4: Detection and mitigation — classifiers + steering interventions
python experiments/scripts/run_phase4.py \
    --config configs/experiment_configs/phase4_interventions.yaml \
    --phase1-run <run_id_from_step1> \
    --phase3-run <run_id_from_step3>
# Outputs to experiments/results/phase4/<run_id>/
```

After each phase completes, open the corresponding notebook for interactive analysis:

| Phase | Script | Notebook |
|-------|--------|----------|
| Datasets | `build_datasets.py` | `notebooks/01_data_exploration.ipynb` |
| Feature discovery | `run_phase1.py` | `notebooks/02_feature_discovery.ipynb` |
| Circuit tracing | `run_phase2.py` | `notebooks/03_circuit_tracing.ipynb` |
| Shared mechanism | `run_phase3.py` | `notebooks/04_transfer_analysis.ipynb` |
| Interventions | `run_phase4.py` | `notebooks/05_interventions.ipynb` |
| Scale analysis | (run Phase 1 on each model) | `notebooks/06_scale_comparison.ipynb` |

Each script accepts `--model` and `--output-dir` to override the config defaults, and `--device` to target a specific GPU. Results are written to `experiments/results/` with timestamped subdirectories.

---

## Prior Work This Extends

**CoT Faithfulness Detection** — Linear probe work achieving 88% accuracy in detecting whether CoT reasoning faithfully reflects the model's internal process. The current project goes from probe-level detection to circuit-level attribution.

**Value-Aligned Confabulation (VAC) Research** — Framework for context-dependent evaluation of factually ungrounded outputs demonstrating that traditional metrics fail to distinguish harmful from beneficial confabulation. The current project provides the mechanistic explanation for output-internal divergence.

**Greater-Than Circuit SAEs** — Hands-on experience with the exact toolchain (JumpReLU SAEs, feature analysis, circuit discovery) that Gemma Scope 2 scales up.

---

## Key Literature

**Core: CoT faithfulness**
- Turpin et al. (2023) — "Language Models Don't Always Say What They Think": biasing features influence CoT without being mentioned
- Arcuschin et al. (2025) — "CoT Reasoning In The Wild Is Not Always Faithful": post-hoc rationalization rates up to 13% on realistic prompts
- METR (2025) — "CoT May Be Highly Informative Despite Unfaithfulness": unfaithfulness concentrates in simple cases, not complex multi-step reasoning
- Barez et al. (2025) — "Chain-of-Thought Is Not Explainability": procedural soundness, causal relevance, completeness as evaluation dimensions

**Core: sycophancy**
- "Sycophancy Is Not One Thing" (2025): sycophantic agreement, genuine agreement, and sycophantic praise are linearly separable along distinct axes
- Rimsky et al. (2024): sycophancy can be steered via DiffMean activation vectors
- "Mitigating Sycophancy via Sparse Autoencoders" (OpenReview 2025): SAF method on Gemma-2-2b-it, layer 17 identified as critical; direct precedent for the current project

**Core: interpretability infrastructure**
- Gemma Scope 2 Technical Report (2025): SAEs + transcoders on all Gemma 3 layers; Matryoshka training
- Anthropic Circuit Tracing (2025): attribution graphs via cross-layer transcoders, applied to Claude 3.5 Haiku
- "Biology of a Large Language Model" (2025): perturbation methodology for circuit validation
- GDM Negative Results for SAEs (2025): SAE probes underperform dense probes for jailbreak detection — motivates circuit-level (not just probe-level) approach

Full reference list is in `paper/references.bib`.

---

## Open Questions

- Should **refusal** be included as a third failure mode? Structurally it is the reverse of sycophancy: the model "knows" it can answer but outputs a refusal. Including it would make this a triptych of override behaviors, but also risks scope expansion.
- How to frame the GDM negative SAE result: our approach is circuit-level attribution, not downstream task probing, but this distinction needs to be argued carefully in the paper.
- Does the override circuit behave differently for multilingual prompts? Relevant for the Deep Learning Indaba connection and potentially a natural extension.
- Does narrow misalignment training (as in the emergent misalignment literature) activate the same override circuits? If so, this project may have broader implications than the CoT/sycophancy framing suggests.

---

## License

MIT. See `LICENSE`.
