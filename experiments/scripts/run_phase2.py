"""
Phase 2: Circuit Tracing

Constructs attribution graphs for prompts where the model demonstrates internal
knowledge of the correct answer but outputs an incorrect or sycophantic response.
Validates candidate override circuits via ablation and activation patching.

Requires Phase 1 outputs: differential feature catalogs per dataset.

Usage:
    python experiments/scripts/run_phase2.py \
        --config configs/experiment_configs/phase2_circuits.yaml \
        --phase1-run <run_id>
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

from src.circuits.attribution import AttributionGraphBuilder
from src.circuits.transcoder_tracing import (
    aggregate_divergence_layers,
    find_consistent_edges,
    find_divergence_layers,
    trace_circuits_for_condition,
)
from src.circuits.validation import CircuitValidator
from src.circuits.visualization import draw_attribution_graph
from src.data.loaders import load_pairs
from src.data.prompt_templates import DatasetType
from src.features.extraction import FeatureExtractor, load_activations, save_activations
from src.utils.gemma_scope import load_all_layer_saes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase2")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Circuit tracing")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase1-run", required=True, help="Phase 1 run_id to load seed features from")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--n-prompts-to-trace",
        type=int,
        default=50,
        help="Number of prompts to trace per condition (circuit tracing is expensive)",
    )
    return parser.parse_args()


def load_phase1_seed_features(
    phase1_results_dir: Path,
    phase1_run_id: str,
    conditions: list[str],
    top_k: int,
) -> dict[str, list[tuple[int, int]]]:
    """
    Load top-k differential features from Phase 1 as (layer, feature_index) seeds.
    """
    seeds: dict[str, list[tuple[int, int]]] = {}
    for condition in conditions:
        # Map condition names to dataset names
        dataset_map = {
            "cot_unfaithful": "cot_bias",
            "sycophancy": "sycophancy_pressure",
            "control": None,
        }
        dataset = dataset_map.get(condition)
        if dataset is None:
            continue

        diff_path = phase1_results_dir / phase1_run_id / dataset / "differential_features.json"
        if not diff_path.exists():
            logger.warning("Phase 1 results not found for condition %s at %s", condition, diff_path)
            continue

        with open(diff_path) as f:
            diff_records = json.load(f)

        # Flatten across layers, sort by |delta|, take top_k
        all_features = []
        for layer_str, features in diff_records.items():
            for feat in features:
                all_features.append((feat["layer"], feat["feature_index"], abs(feat["delta_activation"])))

        all_features.sort(key=lambda x: x[2], reverse=True)
        seeds[condition] = [(layer, fidx) for layer, fidx, _ in all_features[:top_k]]
        logger.info("Loaded %d seed features for condition '%s'", len(seeds[condition]), condition)

    return seeds


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    model_name = args.model or cfg.model
    output_dir = Path(args.output_dir or cfg.output.results_dir)
    phase1_results_dir = Path(cfg.phase1_results)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 2 run %s | model=%s", run_id, model_name)

    model_cfg = OmegaConf.load("configs/models.yaml")
    hf_id = model_cfg.models[model_name].hf_id

    logger.info("Loading model: %s", hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()

    saes = load_all_layer_saes(
        model_name=model_name,
        hook_point="resid_post",
        width_multiplier=16,
        cache_dir="data/activations/sae_weights",
        device=args.device,
    )

    # Load transcoders
    from src.utils.gemma_scope import LazyTranscoderDict, N_LAYERS
    n_layers = N_LAYERS[model_name]
    transcoders = LazyTranscoderDict(
        model_name=model_name,
        transcoder_type="cross_layer",
        cache_dir="data/activations/transcoder_weights",
        device=args.device,
        n_layers=n_layers,
    )

    logger.info("Initialized lazy load mapping for %d transcoders", len(transcoders))

    seed_features = load_phase1_seed_features(
        phase1_results_dir,
        args.phase1_run,
        cfg.conditions,
        top_k=cfg.candidate_features_per_condition,
    )

    graph_builder = AttributionGraphBuilder(
        saes=saes,
        transcoders=transcoders,
        edge_threshold=cfg.circuit_construction.edge_threshold,
        max_hops=cfg.circuit_construction.max_hops,
    )

    extractor = FeatureExtractor(model, tokenizer, saes, device=args.device)
    all_consistent_edges = {}

    for condition in cfg.conditions:
        if condition == "control":
            dataset_name = "cot_bias"
        elif condition == "cot_unfaithful":
            dataset_name = "cot_bias"
        else:
            dataset_name = "sycophancy_pressure"

        pairs = load_pairs(DatasetType(dataset_name))
        prompts = [p.control if condition == "control" else p.treatment for p in pairs]
        prompt_ids = [p.prompt_id for p in pairs]

        cache_path = Path("data/activations") / model_name / f"{condition}_circuit.pkl"
        if cache_path.with_suffix(".pkl.gz").exists():
            acts = load_activations(cache_path)
        else:
            acts = None

        if acts is None:
            logger.info("Extracting activations for condition '%s' (%d prompts)", condition, len(prompts))
            acts = extractor.extract(prompts, batch_size=8)
            save_activations(acts, cache_path)

        seeds = seed_features.get(condition, [])
        if not seeds:
            logger.warning("No seed features for condition '%s', skipping circuit tracing.", condition)
            continue

        logger.info("Tracing circuits for condition '%s'", condition)
        graphs = trace_circuits_for_condition(
            condition_activations=acts,
            seed_features=seeds,
            graph_builder=graph_builder,
            prompt_ids=prompt_ids,
            condition=condition,
            n_prompts_to_trace=args.n_prompts_to_trace,
        )

        consistent_edges = find_consistent_edges(graphs, min_frequency=0.3)
        all_consistent_edges[condition] = consistent_edges

        logger.info(
            "Condition '%s': %d consistent edges (freq >= 0.3)", condition, len(consistent_edges)
        )

        # Save graphs
        condition_dir = output_dir / condition
        condition_dir.mkdir(exist_ok=True)

        edges_serializable = [
            {"source": list(src), "target": list(tgt), "frequency": freq}
            for src, tgt, freq in consistent_edges
        ]
        with open(condition_dir / "consistent_edges.json", "w") as f:
            json.dump(edges_serializable, f, indent=2)

        # Draw a sample graph
        if graphs:
            sample_graph = graphs[0]
            draw_attribution_graph(
                sample_graph,
                output_path=condition_dir / "sample_attribution_graph.png",
                title=f"Attribution graph — {condition} — {sample_graph.prompt_id}",
            )

    # Divergence layer analysis: compare cot_unfaithful vs. control
    if "cot_unfaithful" in seed_features and "control" in seed_features:
        ctrl_acts = load_activations(Path("data/activations") / model_name / "control_circuit.pkl")
        trt_acts = load_activations(Path("data/activations") / model_name / "cot_unfaithful_circuit.pkl")

        if ctrl_acts is None or trt_acts is None:
            logger.warning("Skipping divergence layer analysis — activation cache missing or corrupted.")
        else:
            seed_indices = [fidx for _, fidx in seed_features.get("cot_unfaithful", [])[:20]]

            divergence_results = find_divergence_layers(ctrl_acts, trt_acts, seed_indices)
            divergence_freq = aggregate_divergence_layers(divergence_results)

            peak_div_layer = max(divergence_freq, key=divergence_freq.get)
            logger.info(
                "Divergence layer analysis: peak at layer %d (%.1f%% of prompts)",
                peak_div_layer, 100 * divergence_freq[peak_div_layer],
            )

            with open(output_dir / "divergence_layers.json", "w") as f:
                json.dump(
                    {
                        "divergence_frequency": {str(k): v for k, v in divergence_freq.items()},
                        "peak_layer": peak_div_layer,
                    },
                    f,
                    indent=2,
                )

    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "model": model_name,
                "phase1_run": args.phase1_run,
                "n_transcoders_loaded": len(transcoders),
                "conditions_traced": list(all_consistent_edges.keys()),
                "consistent_edge_counts": {
                    c: len(edges) for c, edges in all_consistent_edges.items()
                },
            },
            f,
            indent=2,
        )

    logger.info("Phase 2 complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
