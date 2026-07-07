"""
Phase 1: Feature Discovery

Runs differential activation analysis across all five datasets on a single model.

Usage:
    python experiments/scripts/run_phase1.py \
        --config configs/experiment_configs/phase1_features.yaml \
        [--model gemma3_4b] \
        [--datasets cot_bias cot_contradiction sycophancy_opinion] \
        [--output-dir experiments/results/phase1]
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

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.data.loaders import load_pairs
from src.data.prompt_templates import DatasetType
from src.features.differential import compute_differential_activation, layer_summary
from src.features.extraction import FeatureExtractor, load_activations, save_activations
from src.utils.gemma_scope import load_all_layer_saes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase1")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Feature discovery")
    parser.add_argument("--config", required=True, help="Path to phase1_features.yaml")
    parser.add_argument("--model", default=None, help="Override model from config")
    parser.add_argument("--datasets", nargs="+", default=None, help="Override dataset list")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    model_name = args.model or cfg.model
    datasets = list(args.datasets or cfg.datasets)
    output_dir = Path(args.output_dir or cfg.output.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Phase 1 run %s | model=%s | datasets=%s", run_id, model_name, datasets)

    # Load model configs
    model_cfg = OmegaConf.load("configs/models.yaml")
    hf_id = model_cfg.models[model_name].hf_id

    logger.info("Loading model: %s", hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()

    sae_cfg = OmegaConf.load("configs/sae_configs.yaml")
    logger.info("Loading SAEs: %dx width", int(cfg.sae_width.replace("x", "")))
    saes = load_all_layer_saes(
        model_name=model_name,
        hook_point=cfg.hook_point,
        width_multiplier=int(cfg.sae_width.replace("x", "")),
        cache_dir="data/activations/sae_weights",
        device=args.device,
    )

    extractor = FeatureExtractor(model, tokenizer, saes, device=args.device)
    all_results = {}

    for dataset_name in datasets:
        logger.info("Processing dataset: %s", dataset_name)
        pairs = load_pairs(DatasetType(dataset_name))
        controls = [p.control for p in pairs]
        treatments = [p.treatment for p in pairs]

        # Check activation cache
        cache_base = Path("data/activations") / model_name / dataset_name
        ctrl_cache = cache_base / "control.pkl"
        trt_cache = cache_base / "treatment.pkl"

        if ctrl_cache.with_suffix(".pkl.gz").exists():
            logger.info("Loading cached control activations from %s", ctrl_cache)
            ctrl_acts = load_activations(ctrl_cache)
        else:
            logger.info("Extracting control activations (%d prompts)", len(controls))
            ctrl_acts = extractor.extract(controls, batch_size=cfg.dataset_params.batch_size)
            save_activations(ctrl_acts, ctrl_cache)

        if trt_cache.with_suffix(".pkl.gz").exists():
            logger.info("Loading cached treatment activations from %s", trt_cache)
            trt_acts = load_activations(trt_cache)
        else:
            logger.info("Extracting treatment activations (%d prompts)", len(treatments))
            trt_acts = extractor.extract(treatments, batch_size=cfg.dataset_params.batch_size)
            save_activations(trt_acts, trt_cache)

        logger.info("Running differential activation analysis")
        differential = compute_differential_activation(
            ctrl_acts,
            trt_acts,
            alpha=cfg.differential_analysis.alpha,
            top_k=cfg.differential_analysis.top_k_features,
        )

        summary = layer_summary(differential)
        all_results[dataset_name] = {
            "layer_summary": summary,
            "n_significant_total": sum(len(v) for v in differential.values()),
        }

        logger.info(
            "Dataset %s: %d total significant features",
            dataset_name,
            all_results[dataset_name]["n_significant_total"],
        )

        # Save per-dataset results
        dataset_out = output_dir / run_id / dataset_name
        dataset_out.mkdir(parents=True, exist_ok=True)

        with open(dataset_out / "layer_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save differential features as JSON for downstream phases
        diff_records = {}
        for layer, features in differential.items():
            diff_records[layer] = [
                {
                    "feature_index": f.feature_index,
                    "layer": f.layer,
                    "delta_activation": f.delta_activation,
                    "t_statistic": f.t_statistic,
                    "p_value_corrected": f.p_value_corrected,
                    "effect_size": f.effect_size,
                    "mean_control": f.mean_control,
                    "mean_treatment": f.mean_treatment,
                }
                for f in features
            ]
        with open(dataset_out / "differential_features.json", "w") as f:
            json.dump(diff_records, f, indent=2)

    # Overall run summary
    with open(output_dir / run_id / "run_summary.json", "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "model": model_name,
                "datasets": datasets,
                "results": all_results,
            },
            f,
            indent=2,
        )

    logger.info("Phase 1 complete. Results in %s/%s", output_dir, run_id)


if __name__ == "__main__":
    main()
