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
