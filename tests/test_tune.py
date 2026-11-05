"""
test_tune.py - Verifies hyperparameter tuning pipeline execution and artifact generation
"""

import json
from pathlib import Path
import subprocess
import sys


def test_tune_pipeline_synthetic_execution(tmp_path):
    output_dir = tmp_path / "tuning_out"
    plots_dir = tmp_path / "tuning_plots"

    cmd = [
        sys.executable,
        "scripts/tune.py",
        "--smoke_test",
        "--epochs_per_trial", "1",
        "--batch_size", "4",
        "--output_dir", str(output_dir),
        "--plots_dir", str(plots_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"tune.py failed:\n{res.stdout}\n{res.stderr}"

    # Verify generated artifacts
    hist_json = output_dir / "tuning_history.json"
    hist_csv = output_dir / "tuning_history.csv"
    best_json = output_dir / "best_hyperparameters.json"
    plot_png = plots_dir / "hyperparameter_trials.png"

    assert hist_json.exists(), "tuning_history.json was not created!"
    assert hist_csv.exists(), "tuning_history.csv was not created!"
    assert best_json.exists(), "best_hyperparameters.json was not created!"
    assert plot_png.exists(), "hyperparameter_trials.png was not created!"

    with open(hist_json) as f:
        records = json.load(f)
    assert len(records) == 4, f"Expected 4 trial records, got {len(records)}"

    with open(best_json) as f:
        best_cfg = json.load(f)
    assert "lr" in best_cfg
    assert "weight_decay" in best_cfg
    assert "dropout" in best_cfg
