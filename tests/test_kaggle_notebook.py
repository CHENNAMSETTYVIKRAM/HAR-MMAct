"""
test_kaggle_notebook.py - Verifies kaggle_mmact_har.ipynb format, cells, and syntax
"""

import ast
import json
from pathlib import Path


def test_kaggle_notebook_structure():
    nb_path = Path("notebooks/kaggle_mmact_har.ipynb")
    assert nb_path.exists(), "Notebook file missing!"

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
        # Filter shell commands and magics
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
                assert False, f"Syntax error in notebook cell {i}: {e}\nCode:\n{code}"


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
