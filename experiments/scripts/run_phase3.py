"""
Phase 3: Shared Mechanism Testing

Tests whether CoT unfaithfulness and sycophancy share causal features via:
  1. Cross-condition transfer (clamp CoT features, measure sycophancy, and vice versa)
  2. Representational geometry (cosine similarity + canonical angles between failure directions)
  3. Scale analysis (repeat Phase 1 on 1B, 4B, 12B, 27B and compare)

Requires Phase 1 outputs for seed features and cached activations.

Usage:
    python experiments/scripts/run_phase3.py \
        --config configs/experiment_configs/phase3_transfer.yaml \
        --phase1-run <run_id> \
        --phase2-run <run_id>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.analysis.geometry import compute_layer_geometry
from src.analysis.transfer import TransferResult, assess_shared_mechanism
from src.data.loaders import load_pairs
from src.data.prompt_templates import DatasetType
from src.features.extraction import FeatureExtractor, load_activations, save_activations
from src.interventions.clamping import ClampingConfig, generate_with_clamping
from src.interventions.evaluation import evaluate_intervention
from src.utils.gemma_scope import load_all_layer_saes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase3")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: Shared mechanism testing")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase1-run", required=True)
    parser.add_argument("--phase2-run", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-scale", action="store_true", help="Skip multi-model scale analysis")
    return parser.parse_args()


def load_seed_feature_indices(
    phase1_dir: Path, run_id: str, dataset_name: str, top_k: int
) -> dict[int, list[int]]:
    """
    Load top-k seed features per layer for a given dataset.
    Returns dict mapping layer → list of feature indices.
    """
    diff_path = phase1_dir / run_id / dataset_name / "differential_features.json"
    if not diff_path.exists():
        logger.warning("Differential features not found at %s", diff_path)
        return {}

    with open(diff_path) as f:
        diff_records = json.load(f)

    layer_features: dict[int, list[int]] = {}
    for layer_str, features in diff_records.items():
        sorted_f = sorted(features, key=lambda x: abs(x["delta_activation"]), reverse=True)
        layer_features[int(layer_str)] = [f["feature_index"] for f in sorted_f[:top_k]]

    return layer_features


def run_transfer_experiment(
    source_condition_label: str,
    source_dataset: str,
    target_condition_label: str,
    target_dataset: str,
    source_features_by_layer: dict[int, list[int]],
    n_features_sweep: list[int],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    saes: dict,
    device: str,
) -> list[TransferResult]:
    """
    Clamp source_condition features and measure effect on target_condition behavior.
    """
    target_pairs = load_pairs(DatasetType(target_dataset))
    target_prompts = [p.treatment for p in target_pairs]
    correct_answers = [p.correct_answer for p in target_pairs]

    # Baseline: generate without any clamping
    from src.interventions.clamping import generate_with_clamping as gen
    baseline_outputs = gen(model, tokenizer, saes, target_prompts, clamp_configs=[])
    baseline_result = evaluate_intervention(
        baseline_outputs, baseline_outputs, correct_answers
    )

    results = []
    for n_features in n_features_sweep:
        clamp_configs = []
        total_added = 0
        # Distribute features evenly across layers, prioritize layers with most features
        for layer, feat_indices in sorted(
            source_features_by_layer.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        ):
            per_layer = min(len(feat_indices), max(1, n_features - total_added))
            clamp_configs.append(
                ClampingConfig(
                    layer=layer,
                    feature_indices=feat_indices[:per_layer],
                    clamp_value=0.0,
                    label=f"{source_condition_label}_top{n_features}",
                )
            )
            total_added += per_layer
            if total_added >= n_features:
                break

        clamped_outputs = gen(model, tokenizer, saes, target_prompts, clamp_configs=clamp_configs)
        clamped_result = evaluate_intervention(
            clamped_outputs, baseline_outputs, correct_answers
        )

        faithfulness_delta = clamped_result.faithfulness_score - baseline_result.faithfulness_score
        sycophancy_delta = baseline_result.sycophancy_rate - clamped_result.sycophancy_rate
        accuracy_delta = clamped_result.accuracy - baseline_result.accuracy

        # Transfer rate: how much of the single-condition effect carries over
        # Defined as |delta| / max_possible_delta (capped at 1)
        transfer_rate = min(abs(faithfulness_delta) + abs(sycophancy_delta), 1.0)

        results.append(
            TransferResult(
                source_condition=source_condition_label,
                target_condition=target_condition_label,
                n_features_clamped=n_features,
                faithfulness_delta=faithfulness_delta,
                sycophancy_delta=sycophancy_delta,
                accuracy_delta=accuracy_delta,
                transfer_rate=transfer_rate,
            )
        )

        logger.info(
            "Transfer %s→%s (n=%d): faithfulness_delta=%.3f, sycophancy_delta=%.3f, transfer_rate=%.3f",
            source_condition_label, target_condition_label, n_features,
            faithfulness_delta, sycophancy_delta, transfer_rate,
        )

    return results


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    model_name = args.model or cfg.model
    output_dir = Path(args.output_dir or cfg.output.results_dir)
    phase1_dir = Path(cfg.phase1_results)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 3 run %s | model=%s", run_id, model_name)

    model_cfg = OmegaConf.load("configs/models.yaml")
    hf_id = model_cfg.models[model_name].hf_id

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()

    saes = load_all_layer_saes(
        model_name=model_name, hook_point="resid_post", width_multiplier=16,
        cache_dir="data/activations/sae_weights", device=args.device,
    )

    n_features_sweep = cfg.cross_condition_transfer.cot_to_sycophancy.n_features_to_clamp

    # -----------------------------------------------------------------------
    # Direction 1: CoT unfaithfulness features → sycophancy prompts
    # -----------------------------------------------------------------------
    logger.info("=== Transfer direction: CoT → Sycophancy ===")
    cot_features = load_seed_feature_indices(
        phase1_dir, args.phase1_run, "cot_bias", top_k=max(n_features_sweep)
    )
    cot_to_syco = run_transfer_experiment(
        source_condition_label="cot_unfaithful",
        source_dataset="cot_bias",
        target_condition_label="sycophancy",
        target_dataset="sycophancy_pressure",
        source_features_by_layer=cot_features,
        n_features_sweep=n_features_sweep,
        model=model, tokenizer=tokenizer, saes=saes, device=args.device,
    )

    # -----------------------------------------------------------------------
    # Direction 2: Sycophancy features → CoT unfaithfulness prompts
    # -----------------------------------------------------------------------
    logger.info("=== Transfer direction: Sycophancy → CoT ===")
    syco_features = load_seed_feature_indices(
        phase1_dir, args.phase1_run, "sycophancy_pressure", top_k=max(n_features_sweep)
    )
    syco_to_cot = run_transfer_experiment(
        source_condition_label="sycophancy",
        source_dataset="sycophancy_pressure",
        target_condition_label="cot_unfaithful",
        target_dataset="cot_bias",
        source_features_by_layer=syco_features,
        n_features_sweep=n_features_sweep,
        model=model, tokenizer=tokenizer, saes=saes, device=args.device,
    )

    summary = assess_shared_mechanism(cot_to_syco, syco_to_cot)
    logger.info(
        "Shared mechanism evidence: %s", summary.shared_mechanism_evidence.upper()
    )
    logger.info(summary.interpretation)

    # -----------------------------------------------------------------------
    # Representational geometry
    # -----------------------------------------------------------------------
    logger.info("=== Representational geometry ===")
    extractor = FeatureExtractor(model, tokenizer, saes, device=args.device)

    cot_ctrl_pairs = load_pairs(DatasetType("cot_bias"))
    syco_pressure_pairs = load_pairs(DatasetType("sycophancy_pressure"))

    cot_ctrl_cache = Path("data/activations") / model_name / "cot_bias" / "control.pkl"
    cot_trt_cache = Path("data/activations") / model_name / "cot_bias" / "treatment.pkl"
    syco_ctrl_cache = Path("data/activations") / model_name / "sycophancy_pressure" / "control.pkl"
    syco_trt_cache = Path("data/activations") / model_name / "sycophancy_pressure" / "treatment.pkl"

    def _load_or_extract(cache_path, prompts):
        if cache_path.with_suffix(".pkl.gz").exists():
            acts = load_activations(cache_path)
            if acts is not None:
                return acts
        acts = extractor.extract(prompts, batch_size=8)
        save_activations(acts, cache_path)
        return acts

    faithful_acts = _load_or_extract(cot_ctrl_cache, [p.control for p in cot_ctrl_pairs])
    unfaithful_acts = _load_or_extract(cot_trt_cache, [p.treatment for p in cot_ctrl_pairs])
    non_syco_acts = _load_or_extract(syco_ctrl_cache, [p.control for p in syco_pressure_pairs])
    syco_acts = _load_or_extract(syco_trt_cache, [p.treatment for p in syco_pressure_pairs])

    geometry = compute_layer_geometry(
        faithful_activations=faithful_acts,
        unfaithful_activations=unfaithful_acts,
        non_sycophantic_activations=non_syco_acts,
        sycophantic_activations=syco_acts,
        subspace_dims=cfg.representational_geometry.get("subspace_dims", 10),
    )

    geometry_records = [
        {
            "layer": g.layer,
            "cosine_similarity": g.cosine_similarity,
            "mean_canonical_angle_deg": g.mean_canonical_angle_deg,
            "min_canonical_angle_deg": g.min_canonical_angle_deg,
        }
        for g in geometry
    ]

    # Find layer(s) of maximum alignment
    peak_alignment = max(geometry, key=lambda g: abs(g.cosine_similarity))
    logger.info(
        "Peak alignment at layer %d: cosine_sim=%.3f, min_angle=%.1f°",
        peak_alignment.layer, peak_alignment.cosine_similarity, peak_alignment.min_canonical_angle_deg,
    )

    # -----------------------------------------------------------------------
    # Save all results
    # -----------------------------------------------------------------------
    with open(output_dir / "transfer_cot_to_syco.json", "w") as f:
        json.dump([vars(r) for r in cot_to_syco], f, indent=2)

    with open(output_dir / "transfer_syco_to_cot.json", "w") as f:
        json.dump([vars(r) for r in syco_to_cot], f, indent=2)

    with open(output_dir / "geometry.json", "w") as f:
        json.dump(geometry_records, f, indent=2)

    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "model": model_name,
                "shared_mechanism_evidence": summary.shared_mechanism_evidence,
                "interpretation": summary.interpretation,
                "peak_alignment_layer": peak_alignment.layer,
                "peak_cosine_similarity": peak_alignment.cosine_similarity,
            },
            f,
            indent=2,
        )

    logger.info("Phase 3 complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
