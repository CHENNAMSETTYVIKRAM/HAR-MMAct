"""
test_kaggle_notebook.py - Verifies kaggle_mmact_har.ipynb format, cells, and syntax
"""

import ast
import json
from pathlib import Path


def test_kaggle_notebook_structure():
    for nb_file in [
        "notebooks/kaggle_mmact_har.ipynb",
        "notebooks/kaggle_mmact_har_benchmark_50ep.ipynb",
        "notebooks/kaggle_mmact_har_complete_fast.ipynb",
    ]:
        nb_path = Path(nb_file)
        assert nb_path.exists(), f"Notebook file {nb_file} missing!"

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        assert "cells" in nb
        assert len(nb["cells"]) >= 9
        assert nb["metadata"]["accelerator"] == "GPU"

        # Verify python syntax for code cells
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            source_lines = cell["source"]
            py_lines = []
            for line in source_lines:
                stripped = line.strip()
                if stripped.startswith(("!", "%", "--", "dest_outputs")):
                    continue
                py_lines.append(line)
            code = "".join(py_lines)
            if code.strip():
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    assert False, f"Syntax error in notebook {nb_file} cell {i}: {e}\nCode:\n{code}"


def test_import_ingestion_pipeline():
    from src.data.ingestion import DatasetIngestionPipeline
    from src.data.indexer import MMActIndexer

    assert DatasetIngestionPipeline is not None
    assert MMActIndexer is not None


def test_no_incompatible_numpy_pins():
    req_path = Path("requirements.txt")
    if req_path.exists():
        content = req_path.read_text()
        assert "<2.0.0" not in content, "requirements.txt still contains strict <2.0.0 pin for numpy!"
        assert "numpy==1.26.4" not in content, "requirements.txt contains strict numpy==1.26.4 pin!"

    kaggle_req_path = Path("requirements-kaggle.txt")
    assert kaggle_req_path.exists(), "requirements-kaggle.txt must exist!"
    kaggle_packages = [
        line.strip().lower()
        for line in kaggle_req_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for pkg in kaggle_packages:
        assert "numpy" not in pkg, f"requirements-kaggle.txt should not reinstall/pin numpy (found {pkg})!"
        assert "pandas" not in pkg, f"requirements-kaggle.txt should not reinstall/pin pandas (found {pkg})!"


def test_kaggle_notebook_preserves_abi():
    nb_path = Path("notebooks/kaggle_mmact_har.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    full_notebook_code = ""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            full_notebook_code += "".join(cell["source"]) + "\n"

    assert "POST-INSTALL ABI & IMPORT VERIFICATION" in full_notebook_code
    assert "requirements-kaggle.txt" in full_notebook_code
    assert "Restart Session" in full_notebook_code
    assert "CORE_C_PACKAGES" in full_notebook_code


def test_kaggle_notebook_multi_dataset_and_symlink_logic():
    nb_path = Path("notebooks/kaggle_mmact_har.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    full_notebook_code = ""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            full_notebook_code += "".join(cell["source"]) + "\n"

    # Step 3 multi-dataset ingestion checks
    assert "MULTI-DATASET INSPECTION & CANONICAL INGESTION" in full_notebook_code
    assert "MMActIndexer.find_files_recursively" in full_notebook_code
    assert "canonical_video" in full_notebook_code
    assert "canonical_sensors" in full_notebook_code
    assert "safe_link_or_copy" in full_notebook_code

    # Step 4 pre-flight audit checks
    assert "PRE-FLIGHT VALIDATION & COMPREHENSIVE MULTI-MODALITY AUDIT" in full_notebook_code
    assert "File Inventory by Source Dataset" in full_notebook_code
    assert "cv2.VideoCapture" in full_notebook_code
    assert "pd.read_csv" in full_notebook_code
    assert "Target Class Coverage" in full_notebook_code
    assert "Zero paired multimodal samples created" in full_notebook_code


def test_multi_dataset_staging_and_symlinks(tmp_path):
    import os
    import shutil
    from src.data.indexer import MMActIndexer
    from src.data.synthetic import generate_synthetic_fixture

    # 1. Create synthetic data
    raw_synthetic = tmp_path / "raw_synthetic"
    generate_synthetic_fixture(raw_synthetic, num_subjects=2, num_cams=1, include_test_anomalies=False)

    # 2. Simulate separate Kaggle dataset inputs
    kaggle_input = tmp_path / "kaggle_input"
    ds1_video_s1 = kaggle_input / "ds1_video_s1"
    ds2_video_s2 = kaggle_input / "ds2_video_s2"
    ds3_sensors = kaggle_input / "ds3_sensors"

    ds1_video_s1.mkdir(parents=True)
    ds2_video_s2.mkdir(parents=True)
    ds3_sensors.mkdir(parents=True)

    # Distribute subjects: subject1 to ds1, subject2 to ds2
    src_v_s1 = raw_synthetic / "trimmed" / "video" / "cross_scene" / "trainval" / "subject1"
    src_v_s2 = raw_synthetic / "trimmed" / "video" / "cross_scene" / "trainval" / "subject2"
    shutil.copytree(src_v_s1, ds1_video_s1 / "subject1")
    shutil.copytree(src_v_s2, ds2_video_s2 / "subject2")

    # Sensors in ds3
    src_sensors = raw_synthetic / "trimmed" / "sensors"
    shutil.copytree(src_sensors, ds3_sensors / "sensors")

    # 3. Simulate canonical staging directory
    canonical_root = tmp_path / "working" / "HAR-MMAct" / "data" / "genuine_mmact"
    can_trimmed = canonical_root / "trimmed"
    can_video = can_trimmed / "video" / "cross_scene" / "trainval"
    can_sensors = can_trimmed / "sensors"

    can_video.mkdir(parents=True)
    can_sensors.mkdir(parents=True)

    # Staging via symlinks (or copy fallback)
    try:
        os.symlink(ds1_video_s1 / "subject1", can_video / "subject1", target_is_directory=True)
        os.symlink(ds2_video_s2 / "subject2", can_video / "subject2", target_is_directory=True)
        for mod_dir in (ds3_sensors / "sensors").iterdir():
            if mod_dir.is_dir():
                os.symlink(mod_dir, can_sensors / mod_dir.name, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation not permitted in this test environment")

    # 4. Verify indexer discovers both subjects and pairs across symlinks
    indexer = MMActIndexer(dataset_root=canonical_root)
    report = indexer.index()

    assert report.total_video_files_found > 0
    assert report.total_sensor_files_found > 0
    assert report.paired_samples_count > 0

    paired_subjects = {s.subject for s in report.paired_samples}
    assert "subject1" in paired_subjects
    assert "subject2" in paired_subjects



def test_abi_import_checks():
    import numpy as np
    import pandas as pd
    import scipy
    import scipy.signal
    import torch

    # Basic C-extension execution
    arr = np.eye(2, dtype=np.float64)
    assert arr.shape == (2, 2)
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert len(df) == 2
    res = scipy.signal.resample([1.0, 2.0, 3.0, 4.0], 8)
    assert len(res) == 8
    t = torch.randn(2, 2)
    t_np = t.numpy()
    assert t_np.shape == (2, 2)


def test_complete_fast_notebook_sections():
    nb_path = Path("notebooks/kaggle_mmact_har_complete_fast.ipynb")
    assert nb_path.exists(), "notebooks/kaggle_mmact_har_complete_fast.ipynb missing!"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    full_code = ""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            full_code += "".join(cell["source"]) + "\n"

    # Verify key requirements are implemented in the fast notebook
    assert "FastMMActStager" in full_code
    assert "staging_report" in full_code
    assert "warm_up_cache" in full_code
    assert "--auto_resume" in full_code
    assert "--skip_if_completed" in full_code
    assert "--benchmark_warmup" in full_code
    assert "best_model_cnn.pth" in full_code
    assert "best_model_tuned_cnn.pth" in full_code
    assert "best_model_mobilenet_v2.pth" in full_code
    assert "best_model_sensor.pth" in full_code
    assert "best_model_multimodal.pth" in full_code
    assert "har_mmact_results.zip" in full_code
    assert "RUN_EXTENDED" in full_code
    assert len(nb["cells"]) >= 30

