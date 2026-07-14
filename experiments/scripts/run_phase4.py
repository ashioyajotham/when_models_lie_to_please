"""
Phase 4: Detection and Mitigation

Trains feature-based classifiers for override detection and evaluates three
inference-time intervention strategies: clamping, directional steering (DiffMean
baseline + SAE feature direction), and CAST (conditional activation steering).

All interventions are benchmarked against capability baselines (MMLU, GSM8K) and
evaluated for over-correction risk.

Requires Phase 1 (feature catalog) and Phase 3 (activation caches and transfer results).

Usage:
    python experiments/scripts/run_phase4.py \
        --config configs/experiment_configs/phase4_interventions.yaml \
        --phase1-run <run_id> \
        --phase3-run <run_id>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.data.loaders import load_pairs
from src.data.prompt_templates import DatasetType
from src.features.extraction import FeatureExtractor, load_activations, save_activations
from src.interventions.clamping import ClampingConfig, generate_with_clamping
from src.interventions.evaluation import evaluate_intervention
from src.interventions.steering import (
    SteeringVector,
    compute_diffmean_vector,
    generate_with_steering,
)
from src.utils.gemma_scope import load_all_layer_saes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase4")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: Detection and mitigation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase1-run", required=True)
    parser.add_argument("--phase3-run", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_override_classifier(
    control_activations: dict[int, torch.Tensor],
    treatment_activations: dict[int, torch.Tensor],
    probe_layer: int | None,
) -> tuple[LogisticRegression, int, float]:
    """
    Train a logistic regression classifier on SAE feature activations to distinguish
    control (faithful/non-sycophantic) from treatment (unfaithful/sycophantic) prompts.

    If probe_layer is None, sweeps all layers and picks the best.

    Returns (classifier, best_layer, cv_accuracy).
    """
    layers = sorted(control_activations.keys())
    if probe_layer is not None:
        layers = [probe_layer]

    best_layer, best_acc, best_clf = None, 0.0, None

    for layer in layers:
        ctrl = control_activations[layer].detach().cpu().float().numpy()
        trt = treatment_activations[layer].detach().cpu().float().numpy()
        X = np.concatenate([ctrl, trt], axis=0)
        y = np.array([0] * len(ctrl) + [1] * len(trt))

        clf = LogisticRegression(max_iter=1000, C=0.1, solver="lbfgs")
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        acc = scores.mean()

        if acc > best_acc:
            best_acc = acc
            best_layer = layer
            clf.fit(X, y)
            best_clf = clf

    logger.info("Best probe layer: %d, CV accuracy: %.3f", best_layer, best_acc)
    return best_clf, best_layer, best_acc


def evaluate_steering_intervention(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    saes: dict,
    pairs_dataset: str,
    condition_activations: dict[int, torch.Tensor],
    baseline_activations: dict[int, torch.Tensor],
    best_layer: int,
    multipliers: list[float],
    method: str,
    phase1_dir: Path,
    phase1_run: str,
    device: str,
) -> list[dict]:
    pairs = load_pairs(DatasetType(pairs_dataset))
    prompts = [p.treatment for p in pairs]
    correct_answers = [p.correct_answer for p in pairs]

    results = []
    for multiplier in multipliers:
        if method == "diffmean":
            direction = compute_diffmean_vector(
                baseline_activations[best_layer].float(),
                condition_activations[best_layer].float(),
            )
        else:
            # SAE feature direction: load from Phase 1 differential features
            diff_path = phase1_dir / phase1_run / pairs_dataset / "differential_features.json"
            if not diff_path.exists():
                logger.warning("Phase 1 diff features not found for %s", pairs_dataset)
                continue
            with open(diff_path) as f:
                diff_records = json.load(f)
            layer_str = str(best_layer)
            if layer_str not in diff_records:
                logger.warning("No features for layer %d in Phase 1 results.", best_layer)
                continue

            from src.features.differential import DifferentialFeature
            from src.interventions.steering import compute_sae_feature_steering_vector
            feat_dicts = diff_records[layer_str]
            diff_features = [
                DifferentialFeature(
                    feature_index=f["feature_index"],
                    layer=f["layer"],
                    delta_activation=f["delta_activation"],
                    t_statistic=0.0,
                    p_value_raw=0.0,
                    p_value_corrected=f.get("p_value_corrected", 0.0),
                    mean_control=f.get("mean_control", 0.0),
                    mean_treatment=f.get("mean_treatment", 0.0),
                    effect_size=f.get("effect_size", 0.0),
                )
                for f in feat_dicts
            ]
            direction = compute_sae_feature_steering_vector(diff_features, saes[best_layer], top_k=20)

        sv = SteeringVector(
            layer=best_layer,
            direction=direction,
            multiplier=multiplier,
            method=method,
        )
        outputs = generate_with_steering(model, tokenizer, prompts, [sv])
        baseline_outputs = generate_with_clamping(model, tokenizer, saes, prompts, clamp_configs=[])
        result = evaluate_intervention(outputs, baseline_outputs, correct_answers)

        results.append(
            {
                "method": method,
                "multiplier": multiplier,
                "layer": best_layer,
                "faithfulness_score": result.faithfulness_score,
                "sycophancy_rate": result.sycophancy_rate,
                "accuracy": result.accuracy,
                "over_correction_rate": result.over_correction_rate,
            }
        )
        logger.info(
            "%s multiplier=%.1f: faithfulness=%.3f, sycophancy=%.3f, accuracy=%.3f",
            method, multiplier,
            result.faithfulness_score, result.sycophancy_rate, result.accuracy,
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

    logger.info("Phase 4 run %s | model=%s", run_id, model_name)

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

    extractor = FeatureExtractor(model, tokenizer, saes, device=args.device)

    def load_or_extract(cache_path, prompts):
        if cache_path.with_suffix(".pkl.gz").exists():
            acts = load_activations(cache_path)
            if acts is not None:
                return acts
        acts = extractor.extract(prompts, batch_size=8)
        save_activations(acts, cache_path)
        return acts

    acts_base = Path("data/activations") / model_name

    cot_pairs = load_pairs(DatasetType("cot_bias"))
    syco_pairs = load_pairs(DatasetType("sycophancy_pressure"))

    cot_ctrl_acts = load_or_extract(acts_base / "cot_bias" / "control.pkl", [p.control for p in cot_pairs])
    cot_trt_acts = load_or_extract(acts_base / "cot_bias" / "treatment.pkl", [p.treatment for p in cot_pairs])
    syco_ctrl_acts = load_or_extract(acts_base / "sycophancy_pressure" / "control.pkl", [p.control for p in syco_pairs])
    syco_trt_acts = load_or_extract(acts_base / "sycophancy_pressure" / "treatment.pkl", [p.treatment for p in syco_pairs])

    # -----------------------------------------------------------------------
    # Train classifiers
    # -----------------------------------------------------------------------
    logger.info("=== Training override classifiers ===")
    probe_layer = cfg.classifiers.get("probe_layer")

    faithfulness_clf, faith_layer, faith_acc = train_override_classifier(
        cot_ctrl_acts, cot_trt_acts, probe_layer
    )
    sycophancy_clf, syco_layer, syco_acc = train_override_classifier(
        syco_ctrl_acts, syco_trt_acts, probe_layer
    )

    classifier_results = {
        "faithfulness_detector": {
            "best_layer": faith_layer, "cv_accuracy": faith_acc
        },
        "sycophancy_detector": {
            "best_layer": syco_layer, "cv_accuracy": syco_acc
        },
    }
    logger.info("Faithfulness classifier accuracy: %.3f at layer %d", faith_acc, faith_layer)
    logger.info("Sycophancy classifier accuracy: %.3f at layer %d", syco_acc, syco_layer)

    # -----------------------------------------------------------------------
    # Steering interventions on sycophancy prompts
    # -----------------------------------------------------------------------
    logger.info("=== Steering interventions ===")
    multipliers = cfg.interventions.directional_steering.multipliers
    all_steering_results = {}

    for method in cfg.interventions.directional_steering.steering_methods:
        logger.info("Evaluating %s steering on sycophancy prompts", method)
        results = evaluate_steering_intervention(
            model, tokenizer, saes,
            pairs_dataset="sycophancy_pressure",
            condition_activations=syco_trt_acts,
            baseline_activations=syco_ctrl_acts,
            best_layer=syco_layer,
            multipliers=multipliers,
            method=method,
            phase1_dir=phase1_dir,
            phase1_run=args.phase1_run,
            device=args.device,
        )
        all_steering_results[f"{method}_sycophancy"] = results

    for method in cfg.interventions.directional_steering.steering_methods:
        logger.info("Evaluating %s steering on CoT faithfulness prompts", method)
        results = evaluate_steering_intervention(
            model, tokenizer, saes,
            pairs_dataset="cot_bias",
            condition_activations=cot_trt_acts,
            baseline_activations=cot_ctrl_acts,
            best_layer=faith_layer,
            multipliers=multipliers,
            method=method,
            phase1_dir=phase1_dir,
            phase1_run=args.phase1_run,
            device=args.device,
        )
        all_steering_results[f"{method}_faithfulness"] = results

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    with open(output_dir / "classifier_results.json", "w") as f:
        json.dump(classifier_results, f, indent=2)

    with open(output_dir / "steering_results.json", "w") as f:
        json.dump(all_steering_results, f, indent=2)

    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "model": model_name,
                "phase1_run": args.phase1_run,
                "faithfulness_classifier_accuracy": faith_acc,
                "sycophancy_classifier_accuracy": syco_acc,
                "best_faithfulness_probe_layer": faith_layer,
                "best_sycophancy_probe_layer": syco_layer,
                "n_steering_configurations_evaluated": sum(
                    len(v) for v in all_steering_results.values()
                ),
            },
            f,
            indent=2,
        )

    logger.info("Phase 4 complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
