"""End-to-end tests for the entire 4-phase research pipeline in mock mode."""

import os
import sys
from pathlib import Path
from unittest.mock import patch
import pytest


@pytest.fixture(scope="module", autouse=True)
def setup_mock_env():
    os.environ["MOCK_PIPELINE"] = "true"
    # Build a tiny dataset for testing if it doesn't exist
    from src.data.dataset_builder import DatasetBuilder
    if not Path("data/processed/cot_bias/pairs.jsonl").exists():
        builder = DatasetBuilder(output_dir="data/processed", min_pairs=5)
        builder.build_all()
    yield
    if "MOCK_PIPELINE" in os.environ:
        del os.environ["MOCK_PIPELINE"]


def test_end_to_end_mock_pipeline(tmp_path):
    results_dir = tmp_path / "results"

    # -------------------------------------------------------------------------
    # Phase 1: Feature Discovery
    # -------------------------------------------------------------------------
    p1_args = [
        "run_phase1.py",
        "--config",
        "configs/experiment_configs/phase1_features.yaml",
        "--output-dir",
        str(results_dir / "phase1"),
        "--model",
        "gemma3_4b",
        "--datasets",
        "cot_bias",
        "sycophancy_pressure",
    ]
    with patch.object(sys, "argv", p1_args):
        # Delete run_phase1 from sys.modules to prevent caching issues if imported elsewhere
        if "experiments.scripts.run_phase1" in sys.modules:
            del sys.modules["experiments.scripts.run_phase1"]
        from experiments.scripts.run_phase1 import main as run_p1
        run_p1()

    p1_runs = list((results_dir / "phase1").iterdir())
    assert len(p1_runs) == 1
    run_id = p1_runs[0].name
    assert (results_dir / "phase1" / run_id / "run_summary.json").exists()
    assert (results_dir / "phase1" / run_id / "cot_bias" / "differential_features.json").exists()

    # -------------------------------------------------------------------------
    # Phase 2: Circuit Tracing
    # -------------------------------------------------------------------------
    p2_args = [
        "run_phase2.py",
        "--config",
        "configs/experiment_configs/phase2_circuits.yaml",
        "--output-dir",
        str(results_dir / "phase2"),
        "--phase1-run",
        run_id,
        "--n-prompts-to-trace",
        "2",
    ]

    # Patch OmegaConf.load to point configurations to our temporary results directory
    from omegaconf import OmegaConf
    original_load = OmegaConf.load

    def patched_load(path, *args, **kwargs):
        c = original_load(path, *args, **kwargs)
        if "phase1_results" in c:
            c.phase1_results = str(results_dir / "phase1")
        if "phase3_results" in c:
            c.phase3_results = str(results_dir / "phase3")
        return c

    with patch("omegaconf.OmegaConf.load", side_effect=patched_load):
        with patch.object(sys, "argv", p2_args):
            if "experiments.scripts.run_phase2" in sys.modules:
                del sys.modules["experiments.scripts.run_phase2"]
            from experiments.scripts.run_phase2 import main as run_p2
            run_p2()

    p2_runs = list((results_dir / "phase2").iterdir())
    assert len(p2_runs) == 1
    p2_run_id = p2_runs[0].name
    assert (results_dir / "phase2" / p2_run_id / "run_summary.json").exists()

    # -------------------------------------------------------------------------
    # Phase 3: Shared Mechanism Testing
    # -------------------------------------------------------------------------
    p3_args = [
        "run_phase3.py",
        "--config",
        "configs/experiment_configs/phase3_transfer.yaml",
        "--output-dir",
        str(results_dir / "phase3"),
        "--phase1-run",
        run_id,
        "--skip-scale",
    ]
    with patch("omegaconf.OmegaConf.load", side_effect=patched_load):
        with patch.object(sys, "argv", p3_args):
            if "experiments.scripts.run_phase3" in sys.modules:
                del sys.modules["experiments.scripts.run_phase3"]
            from experiments.scripts.run_phase3 import main as run_p3
            run_p3()

    p3_runs = list((results_dir / "phase3").iterdir())
    assert len(p3_runs) == 1
    p3_run_id = p3_runs[0].name
    assert (results_dir / "phase3" / p3_run_id / "run_summary.json").exists()

    # -------------------------------------------------------------------------
    # Phase 4: Detection & Mitigation
    # -------------------------------------------------------------------------
    p4_args = [
        "run_phase4.py",
        "--config",
        "configs/experiment_configs/phase4_interventions.yaml",
        "--output-dir",
        str(results_dir / "phase4"),
        "--phase1-run",
        run_id,
        "--phase3-run",
        p3_run_id,
    ]
    with patch("omegaconf.OmegaConf.load", side_effect=patched_load):
        with patch.object(sys, "argv", p4_args):
            if "experiments.scripts.run_phase4" in sys.modules:
                del sys.modules["experiments.scripts.run_phase4"]
            from experiments.scripts.run_phase4 import main as run_p4
            run_p4()

    p4_runs = list((results_dir / "phase4").iterdir())
    assert len(p4_runs) == 1
    p4_run_id = p4_runs[0].name
    assert (results_dir / "phase4" / p4_run_id / "run_summary.json").exists()
