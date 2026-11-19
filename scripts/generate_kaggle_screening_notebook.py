import json
import ast
from pathlib import Path

def make_cell(cell_type, source):
    if isinstance(source, str):
        lines = [l + "\n" for l in source.split("\n")]
        if lines:
            lines[-1] = lines[-1][:-1]
    else:
        lines = source
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

cells = []

# Cell 0: Title and Documentation
cells.append(make_cell("markdown", """# HAR-MMAct — Genuine MMAct Video Model Screening Experiments

**Repository:** `CHENNAMSETTYVIKRAM/HAR-MMAct`  
**Target Branch:** `accuracy-improvements`  
**Scope:** Controlled validation-only screening for Custom CNN and MobileNetV2 video architectures on genuine MMAct data.  

### Strict Experimental & Isolation Safeguards:
1. **Genuine MMAct Data Only:** Synthetic fixtures are strictly barred from discovery and execution.
2. **Absolute Pre-Indexing Test Isolation:** Screening uses train subjects 1–14 and validation subjects 15–17 only. Test subjects 18–20 are excluded **before** directory traversal, video staging, and sample-record construction. Test samples are never constructed, iterated over, loaded, evaluated, or accessed in any way during screening.
3. **One Canonical Train/Validation Manifest:** A single pre-validated train manifest (3,795 samples, subjects 1–14) and validation manifest (780 samples, subjects 15–17) are generated, verified against the original benchmark manifest, and consumed by all five training runs. Every run's sample IDs are recorded and verified for exact 100% ID equality against the canonical manifests.
4. **Verified Transfer Learning Before Execution:** MobileNetV2 requires verified ImageNet pretrained weight loading prior to launching any training runs. Silent fallback to random initialization is strictly forbidden.
5. **Pure Video-Only Input Isolation:** Models receive only video tensor inputs; sensor tensors are never moved to GPU or passed to the model.
6. **Mandatory GPU Execution:** CUDA GPU accelerator is enforced by assertion. CPU execution is strictly prohibited.
7. **Validation Macro-F1 Checkpoint & Winner Selection:** Checkpoint and candidate winner selection strictly use full-precision validation Macro-F1 on the identical 780-sample subject-independent validation split."""))

# Cell 1: Environment, GPU Inspection & Deterministic Random Seeds
cells.append(make_cell("code", """import os
import sys
import subprocess
import torch
import numpy as np

print("=" * 78)
print(" SYSTEM, GPU RUNTIME INSPECTION & DETERMINISTIC SEEDS")
print("=" * 78)
print(f"Python Executable : {sys.executable}")
print(f"PyTorch Version   : {torch.__version__}")
print(f"CUDA Available    : {torch.cuda.is_available()}")

# Hard execution gate: CUDA GPU required
assert torch.cuda.is_available(), (
    "CRITICAL: CUDA GPU accelerator is strictly required for genuine MMAct screening! "
    "CPU training is disallowed. Please enable GPU accelerator under "
    "Kaggle Notebook Settings -> Accelerator -> GPU P100 / T4."
)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"GPU Accelerator   : {gpu_name} ({gpu_mem:.2f} GB VRAM)")
subprocess.run(["nvidia-smi"], check=False)

# Configure deterministic seeds across all libraries
SEED = 42
from src.config import set_seed
set_seed(SEED)
print(f"Global deterministic random seeds initialized: Python={SEED}, NumPy={SEED}, PyTorch={SEED}, CUDA={SEED}")"""))

# Cell 2: Repository Checkout & Code Synchronization
cells.append(make_cell("code", """import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/kaggle/working/HAR-MMAct")
REPO_URL = "https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct.git"
TARGET_BRANCH = "accuracy-improvements"

print(f"[Step 1] Synchronizing repository on branch: {TARGET_BRANCH}...")
if not PROJECT_ROOT.exists():
    print(f"Cloning {REPO_URL} (branch: {TARGET_BRANCH})...")
    subprocess.run(["git", "clone", "--branch", TARGET_BRANCH, REPO_URL, str(PROJECT_ROOT)], check=True)
else:
    print(f"Repository exists at {PROJECT_ROOT}. Fetching latest {TARGET_BRANCH}...")
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "fetch", "origin", TARGET_BRANCH], check=True)
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "checkout", TARGET_BRANCH], check=True)
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "pull", "--ff-only", "origin", TARGET_BRANCH], check=True)

commit_hash = subprocess.check_output(
    ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"]
).decode().strip()
print(f"Verified Active Commit : {commit_hash} on {TARGET_BRANCH}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Install lightweight Kaggle dependencies (dependency failure is fatal)
req_file = PROJECT_ROOT / "requirements-kaggle.txt"
if req_file.exists():
    print(f"Installing dependencies from {req_file}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], check=True)"""))

# Cell 3: Genuine Dataset Discovery & Canonical Staging with Test Subject Pruning
cells.append(make_cell("code", """from pathlib import Path
from src.data.stager import FastMMActStager, discover_kaggle_input_roots

kaggle_input = Path("/kaggle/input")
canonical_data_dir = PROJECT_ROOT / "data" / "genuine_mmact"

print("[Step 2] Discovering genuine MMAct datasets under /kaggle/input (pruning test subjects 18-20)...")
input_roots = discover_kaggle_input_roots(kaggle_input)

# Safety check: Ensure no synthetic directories are included in genuine discovery
for r in input_roots:
    assert "synthetic" not in str(r).lower(), f"CRITICAL ERROR: Synthetic directory detected in genuine input roots: {r}"

print(f"Discovered {len(input_roots)} candidate input roots:")
for r in input_roots:
    print(f"  * {r}")

# Stage dataset while strictly excluding test subjects 18-20 from staging/copying
stager = FastMMActStager(
    canonical_dir=canonical_data_dir,
    input_roots=input_roots,
    max_scan_depth=4,
    excluded_subjects={"subject18", "subject19", "subject20"},
    allowed_subjects={f"subject{i}" for i in range(1, 18)},
)

staging_report = stager.discover_and_stage()
print(staging_report.summary())

# Rigorous staging gates
assert staging_report.video_count > 0, "CRITICAL: No video files found in staged dataset!"
assert len(staging_report.duplicate_conflicts) == 0, f"CRITICAL: Duplicate conflicts detected: {staging_report.duplicate_conflicts}"
assert not any(s in ["subject18", "subject19", "subject20"] for s in staging_report.video_subjects), (
    "CRITICAL: Test subjects detected in staged video subjects!"
)
print("\\n>>> STAGING GATE PASSED: Genuine MMAct dataset successfully staged without synthetic contamination or test subjects.")"""))

# Cell 4: Benchmark Split Consistency & Canonical Manifest Generation
cells.append(make_cell("code", """import json
from pathlib import Path
from src.config import VERIFIED_TARGET_CLASSES, TARGET_CLASS_TO_IDX
from src.data.indexer import MMActIndexer

RESULTS_ROOT = Path("/kaggle/working/HAR-MMAct-results")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

print("[Step 3] Auditing genuine dataset index, verifying benchmark split consistency, and generating canonical manifests...")

# Verify against original benchmark manifest
original_manifest_path = PROJECT_ROOT / "outputs" / "mobilenet_v2" / "experiment_manifest.json"
assert original_manifest_path.exists(), f"CRITICAL: Original benchmark manifest not found at {original_manifest_path}"
benchmark_manifest = json.loads(original_manifest_path.read_text())

expected_train_count = benchmark_manifest["train_samples_count"]  # 3,795
expected_val_count = benchmark_manifest["val_samples_count"]      # 780
print(f"Original Benchmark Expectations: Train={expected_train_count}, Val={expected_val_count}")

# Index with pre-indexing pruning of test subjects 18-20
indexer = MMActIndexer(
    dataset_root=canonical_data_dir,
    required_sensor_modalities=[],
    excluded_subjects={"subject18", "subject19", "subject20"},
    allowed_subjects={f"subject{i}" for i in range(1, 18)},
)
report = indexer.index()

# Separate into train (subjects 1-14) and val (subjects 15-17)
canonical_train_samples = [
    s for s in report.paired_samples 
    if int(s.subject.replace("subject", "")) <= 14
]
canonical_val_samples = [
    s for s in report.paired_samples 
    if 15 <= int(s.subject.replace("subject", "")) <= 17
]

print("=" * 78)
print(f"Total Pre-Filtered Samples  : {len(report.paired_samples)}")
print(f"Train Split (Subjects 1-14) : {len(canonical_train_samples)} samples (Expected: {expected_train_count})")
print(f"Val Split   (Subjects 15-17): {len(canonical_val_samples)} samples (Expected: {expected_val_count})")
print(f"Test Split  (Subjects 18-20): STRICTLY OMITTED & NEVER CONSTRUCTED")
print("=" * 78)

# Strict Split Consistency and Isolation Asserts
assert len(canonical_train_samples) == expected_train_count, (
    f"CRITICAL: Train sample count mismatch! Found {len(canonical_train_samples)}, expected {expected_train_count}. "
    f"Stopping to prevent training on inconsistent split."
)
assert len(canonical_val_samples) == expected_val_count, (
    f"CRITICAL: Validation sample count mismatch! Found {len(canonical_val_samples)}, expected {expected_val_count}. "
    f"Stopping to prevent validation on inconsistent split."
)

for s in canonical_train_samples:
    sub_num = int(s.subject.replace("subject", ""))
    assert 1 <= sub_num <= 14, f"Train sample from invalid subject: {s.subject}"

for s in canonical_val_samples:
    sub_num = int(s.subject.replace("subject", ""))
    assert 15 <= sub_num <= 17, f"Val sample from invalid subject: {s.subject}"

canonical_train_ids = [s.sample_id for s in canonical_train_samples]
canonical_val_ids = [s.sample_id for s in canonical_val_samples]
assert len(set(canonical_train_ids).intersection(set(canonical_val_ids))) == 0, "CRITICAL ERROR: Data leakage between train and val splits!"

train_classes = {s.action for s in canonical_train_samples}
val_classes = {s.action for s in canonical_val_samples}
assert len(train_classes) == 10, f"Expected all 10 target classes in train split, found {len(train_classes)}: {train_classes}"
assert len(val_classes) == 10, f"Expected all 10 target classes in validation split, found {len(val_classes)}: {val_classes}"

# Emit Single Canonical Manifests to be consumed identically by all five training runs
TRAIN_MANIFEST_PATH = RESULTS_ROOT / "canonical_train_manifest.json"
VAL_MANIFEST_PATH = RESULTS_ROOT / "canonical_val_manifest.json"

with open(TRAIN_MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump([s.to_dict() for s in canonical_train_samples], f, indent=2)
with open(VAL_MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump([s.to_dict() for s in canonical_val_samples], f, indent=2)

print(f"Saved canonical train manifest ({len(canonical_train_samples)} samples) to: {TRAIN_MANIFEST_PATH}")
print(f"Saved canonical val manifest   ({len(canonical_val_samples)} samples) to: {VAL_MANIFEST_PATH}")"""))

# Cell 5: MobileNetV2 Pre-Flight Pretrained Verification
cells.append(make_cell("code", """from src.models.video_branch import MobileNetV2VideoBranch

print("=" * 78)
print(" MOBILENETV2 PRETRAINED WEIGHT LOADING PREFLIGHT VERIFICATION")
print("=" * 78)

# Strict preflight verification before ANY experiments are launched
test_branch = MobileNetV2VideoBranch(pretrained=True, require_pretrained=True)
assert test_branch.pretrained_loaded, (
    f"CRITICAL: MobileNetV2 pretrained weights failed to load! Source: {test_branch.pretrained_source}. "
    f"Silent fallback to random initialization is prohibited. Aborting before launching runs."
)
assert test_branch.pretrained_source in ["torchvision-default", "torchvision-legacy", "pure-pytorch-external"], (
    f"Unexpected pretrained source: {test_branch.pretrained_source}"
)
total_params = sum(p.numel() for p in test_branch.parameters())
trainable_params = sum(p.numel() for p in test_branch.parameters() if p.requires_grad)

print(f"Pretrained Source       : {test_branch.pretrained_source}")
print(f"Pretrained Verified     : {test_branch.pretrained_loaded}")
print(f"Normalization Mean/Std  : {test_branch.normalization_mean} / {test_branch.normalization_std}")
print(f"Total Parameters        : {total_params:,}")
print(f"Trainable Parameters    : {trainable_params:,}")
print("Pretrained MobileNetV2 weights verified. Ready to execute screening runs.\\n")"""))

# Cell 6: Custom CNN Screening Experiments (Consuming Canonical Manifests)
cells.append(make_cell("code", """import subprocess
from pathlib import Path

cnn_epochs = 12
batch_size = 16 if torch.cuda.is_available() else 8

cnn_trials = [
    {
        "name": "cnn_dropout035_baseline",
        "desc": "Baseline Custom CNN: dropout 0.35 + inverse-frequency class weights (spatial dropout 0.15/0.20)",
        "output_dir": RESULTS_ROOT / "cnn_dropout035_baseline",
        "cmd": [
            sys.executable, "scripts/train.py",
            "--train_manifest", str(TRAIN_MANIFEST_PATH),
            "--val_manifest", str(VAL_MANIFEST_PATH),
            "--mode", "video_only",
            "--video_backbone", "custom_cnn",
            "--epochs", str(cnn_epochs),
            "--batch_size", str(batch_size),
            "--lr", "1e-3",
            "--dropout", "0.35",
            "--spatial_dropout1", "0.15",
            "--spatial_dropout2", "0.20",
            "--use_class_weights",
            "--seed", "42",
            "--no_test_split",
            "--benchmark_warmup",
            "--output_dir", str(RESULTS_ROOT / "cnn_dropout035_baseline"),
        ]
    },
    {
        "name": "cnn_dropout020_no_cw",
        "desc": "Reduced Regularization: dropout 0.20 without class weights (spatial dropout 0.10/0.15)",
        "output_dir": RESULTS_ROOT / "cnn_dropout020_no_cw",
        "cmd": [
            sys.executable, "scripts/train.py",
            "--train_manifest", str(TRAIN_MANIFEST_PATH),
            "--val_manifest", str(VAL_MANIFEST_PATH),
            "--mode", "video_only",
            "--video_backbone", "custom_cnn",
            "--epochs", str(cnn_epochs),
            "--batch_size", str(batch_size),
            "--lr", "1e-3",
            "--dropout", "0.20",
            "--spatial_dropout1", "0.10",
            "--spatial_dropout2", "0.15",
            "--no_class_weights",
            "--seed", "42",
            "--no_test_split",
            "--benchmark_warmup",
            "--output_dir", str(RESULTS_ROOT / "cnn_dropout020_no_cw"),
        ]
    },
    {
        "name": "cnn_dropout025_candidate",
        "desc": "Candidate Configuration: dropout 0.25 + class weights (spatial dropout 0.10/0.15)",
        "output_dir": RESULTS_ROOT / "cnn_dropout025_candidate",
        "cmd": [
            sys.executable, "scripts/train.py",
            "--train_manifest", str(TRAIN_MANIFEST_PATH),
            "--val_manifest", str(VAL_MANIFEST_PATH),
            "--mode", "video_only",
            "--video_backbone", "custom_cnn",
            "--epochs", str(cnn_epochs),
            "--batch_size", str(batch_size),
            "--lr", "1e-3",
            "--dropout", "0.25",
            "--spatial_dropout1", "0.10",
            "--spatial_dropout2", "0.15",
            "--use_class_weights",
            "--seed", "42",
            "--no_test_split",
            "--benchmark_warmup",
            "--output_dir", str(RESULTS_ROOT / "cnn_dropout025_candidate"),
        ]
    },
]

for trial in cnn_trials:
    print("\\n" + "=" * 78)
    print(f"RUNNING SCREENING EXPERIMENT: {trial['name']}")
    print(f"Description: {trial['desc']}")
    print("=" * 78)
    subprocess.run(trial["cmd"], cwd=str(PROJECT_ROOT), check=True)"""))

# Cell 7: MobileNetV2 Screening Experiments (Consuming Canonical Manifests)
cells.append(make_cell("code", """mobilenet_epochs = 12

mobilenet_trials = [
    {
        "name": "mobilenet_frozen_baseline",
        "desc": "Controlled Frozen Backbone: ImageNet weights + frozen backbone + dropout 0.20 + LR 1e-4",
        "output_dir": RESULTS_ROOT / "mobilenet_frozen_baseline",
        "cmd": [
            sys.executable, "scripts/train.py",
            "--train_manifest", str(TRAIN_MANIFEST_PATH),
            "--val_manifest", str(VAL_MANIFEST_PATH),
            "--mode", "video_only",
            "--video_backbone", "mobilenet_v2",
            "--epochs", str(mobilenet_epochs),
            "--batch_size", str(batch_size),
            "--lr", "1e-4",
            "--dropout", "0.20",
            "--freeze_video",
            "--use_class_weights",
            "--seed", "42",
            "--no_test_split",
            "--require_pretrained",
            "--benchmark_warmup",
            "--output_dir", str(RESULTS_ROOT / "mobilenet_frozen_baseline"),
        ]
    },
    {
        "name": "mobilenet_staged_candidate",
        "desc": "Controlled Staged Fine-Tuning: unfreeze backbone at epoch 4 @ 1e-5, head @ 2e-4",
        "output_dir": RESULTS_ROOT / "mobilenet_staged_candidate",
        "cmd": [
            sys.executable, "scripts/train.py",
            "--train_manifest", str(TRAIN_MANIFEST_PATH),
            "--val_manifest", str(VAL_MANIFEST_PATH),
            "--mode", "video_only",
            "--video_backbone", "mobilenet_v2",
            "--epochs", str(mobilenet_epochs),
            "--batch_size", str(batch_size),
            "--lr", "2e-4",
            "--dropout", "0.20",
            "--staged_finetune",
            "--unfreeze_epoch", "4",
            "--unfreeze_lr", "1e-5",
            "--use_class_weights",
            "--seed", "42",
            "--no_test_split",
            "--require_pretrained",
            "--benchmark_warmup",
            "--output_dir", str(RESULTS_ROOT / "mobilenet_staged_candidate"),
        ]
    },
]

for trial in mobilenet_trials:
    print("\\n" + "=" * 78)
    print(f"RUNNING SCREENING EXPERIMENT: {trial['name']}")
    print(f"Description: {trial['desc']}")
    print("=" * 78)
    subprocess.run(trial["cmd"], cwd=str(PROJECT_ROOT), check=True)"""))

# Cell 8: Strict Validation Metrics Audit & Sample ID Equality Verification
cells.append(make_cell("code", """import json
import math
import pandas as pd
from pathlib import Path

all_trials = cnn_trials + mobilenet_trials
records = []

print("=" * 80)
print(" GENUINE MMAct SCREENING VALIDATION AUDIT (Validation Split Only: 780 Samples)")
print("=" * 80)

# All five experiment outputs are strictly required before proceeding
for t in all_trials:
    out_dir = t["output_dir"]
    summary_path = out_dir / "training_summary.json"
    manifest_path = out_dir / "experiment_manifest.json"
    history_path = out_dir / "history.json"
    train_ids_path = out_dir / "actual_train_sample_ids.json"
    val_ids_path = out_dir / "actual_val_sample_ids.json"
    
    assert summary_path.exists(), f"CRITICAL: Missing training summary for required experiment: {t['name']}"
    assert manifest_path.exists(), f"CRITICAL: Missing experiment manifest for required experiment: {t['name']}"
    assert history_path.exists(), f"CRITICAL: Missing history file for required experiment: {t['name']}"
    assert train_ids_path.exists(), f"CRITICAL: Missing actual train sample IDs file for {t['name']}"
    assert val_ids_path.exists(), f"CRITICAL: Missing actual val sample IDs file for {t['name']}"
        
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    history = json.loads(history_path.read_text())
    
    # 1. Exact Canonical Sample ID Equality Verification
    run_train_ids = json.loads(train_ids_path.read_text())
    run_val_ids = json.loads(val_ids_path.read_text())
    assert run_train_ids == canonical_train_ids, f"CRITICAL: Run {t['name']} train sample IDs differ from canonical manifest!"
    assert run_val_ids == canonical_val_ids, f"CRITICAL: Run {t['name']} val sample IDs differ from canonical manifest!"
    assert len(run_train_ids) == 3795, f"CRITICAL: Expected exactly 3,795 train sample IDs, got {len(run_train_ids)}"
    assert len(run_val_ids) == 780, f"CRITICAL: Expected exactly 780 val sample IDs, got {len(run_val_ids)}"
    
    # 2. Confirm exactly 12 epochs completed
    train_loss_list = history.get("train_loss", [])
    val_loss_list = history.get("val_loss", [])
    val_f1_list = history.get("val_f1", [])
    assert len(train_loss_list) == 12, (
        f"CRITICAL: {t['name']} completed only {len(train_loss_list)}/12 epochs! Incomplete runs are disallowed."
    )
    assert len(val_loss_list) == 12, f"CRITICAL: Expected 12 val_loss entries, got {len(val_loss_list)} in {t['name']}"
    assert len(val_f1_list) == 12, f"CRITICAL: Expected 12 val_f1 entries, got {len(val_f1_list)} in {t['name']}"
    
    # 3. Reject missing or non-finite metric fields (no replacement with zero)
    best_m = summary.get("best_metrics", {})
    for m_field in ["macro_f1", "accuracy", "loss"]:
        assert m_field in best_m, f"CRITICAL: '{m_field}' missing in best_metrics for {t['name']}"
        val = best_m[m_field]
        assert isinstance(val, (int, float)) and math.isfinite(val), f"CRITICAL: Non-finite {m_field} ({val}) in {t['name']}"
    
    # 4. Derive best epoch from maximum validation Macro-F1 and verify against recorded best_epoch
    max_val_f1 = max(val_f1_list)
    derived_best_epoch = val_f1_list.index(max_val_f1) + 1
    recorded_best_epoch = summary.get("best_epoch") or best_m.get("epoch")
    assert recorded_best_epoch == derived_best_epoch, (
        f"CRITICAL: Best epoch discrepancy in {t['name']}! Recorded {recorded_best_epoch}, but max val_f1 is at epoch {derived_best_epoch}"
    )
    
    raw_macro_f1 = float(best_m["macro_f1"])  # Full-precision unrounded float
    
    records.append({
        "Experiment Name": t["name"],
        "Architecture": manifest.get("video_backbone", "unknown"),
        "Dropout": manifest.get("dropout", "N/A"),
        "Spatial Drop": f"{manifest.get('spatial_dropout1')}/{manifest.get('spatial_dropout2')}" if manifest.get("spatial_dropout1") else "N/A",
        "Learning Rate": manifest.get("lr", "N/A"),
        "Class Weights": "Yes" if "--use_class_weights" in t["cmd"] else "No",
        "Frozen Backbone": manifest.get("freeze_video", False),
        "Staged Finetune": manifest.get("staged_finetune", False),
        "Epochs Completed": len(train_loss_list),
        "Best Epoch": derived_best_epoch,
        "Best Val Macro-F1": round(raw_macro_f1, 4),
        "raw_macro_f1": raw_macro_f1,
        "Val Accuracy (%)": round(best_m["accuracy"] * 100.0, 2),
        "Best Val Loss": round(best_m["loss"], 4),
        "Train Samples": manifest.get("train_samples_count", 0),
        "Val Samples": manifest.get("val_samples_count", 0),
        "Sample IDs Match": True,
        "Runtime (s)": round(summary.get("runtime_seconds", 0.0), 1),
    })

    # Display per-class metrics (strictly requiring valid precision, recall, f1, support)
    per_class = best_m.get("per_class", {})
    assert per_class, f"CRITICAL: per_class metrics missing in {t['name']}"
    print(f"\\n--- Per-Class Performance Breakdown: {t['name']} ---")
    pc_rows = []
    for cname, cmetrics in per_class.items():
        if isinstance(cmetrics, dict):
            for field in ["precision", "recall", "f1-score", "support"]:
                assert field in cmetrics and math.isfinite(cmetrics[field]), f"CRITICAL: Non-finite {field} for {cname} in {t['name']}"
            pc_rows.append({
                "Class": cname,
                "Precision": round(cmetrics["precision"], 4),
                "Recall": round(cmetrics["recall"], 4),
                "F1-Score": round(cmetrics["f1-score"], 4),
                "Support": cmetrics["support"],
            })
    if pc_rows:
        print(pd.DataFrame(pc_rows).to_string(index=False))

df = pd.DataFrame(records)
print("\\n" + "=" * 80)
print(" CONSOLIDATED SCREENING SUMMARY")
print("=" * 80)
display_cols = [c for c in df.columns if c != "raw_macro_f1"]
print(df[display_cols].to_string(index=False))

summary_csv = RESULTS_ROOT / "screening_genuine_validation_summary.csv"
df[display_cols].to_csv(summary_csv, index=False)
print(f"\\nSaved consolidated screening audit table to: {summary_csv}")"""))

# Cell 9: Candidate Selection Gate (Full-Precision Validation Macro-F1)
cells.append(make_cell("code", """print("=" * 80)
print(" CANDIDATE SELECTION DECISION (Based Strictly on Full-Precision Validation Macro-F1)")
print("=" * 80)

assert len(records) == 5, f"CRITICAL: Expected exactly 5 screening records, found {len(records)}"

# Filter Custom CNN runs
cnn_runs = [r for r in records if "cnn" in r["Architecture"]]
assert len(cnn_runs) == 3, f"Expected 3 Custom CNN runs, found {len(cnn_runs)}"
best_cnn = max(cnn_runs, key=lambda x: x["raw_macro_f1"])
print(f"Best Custom CNN Candidate : {best_cnn['Experiment Name']}")
print(f"  * Validation Macro-F1   : {best_cnn['raw_macro_f1']:.6f} (display: {best_cnn['Best Val Macro-F1']})")
print(f"  * Validation Accuracy   : {best_cnn['Val Accuracy (%)']}%")
print(f"  * Best Validation Loss  : {best_cnn['Best Val Loss']}")

# Filter MobileNetV2 runs
mob_runs = [r for r in records if "mobilenet" in r["Architecture"]]
assert len(mob_runs) == 2, f"Expected 2 MobileNetV2 runs, found {len(mob_runs)}"
best_mob = max(mob_runs, key=lambda x: x["raw_macro_f1"])
print(f"\\nBest MobileNetV2 Candidate: {best_mob['Experiment Name']}")
print(f"  * Validation Macro-F1   : {best_mob['raw_macro_f1']:.6f} (display: {best_mob['Best Val Macro-F1']})")
print(f"  * Validation Accuracy   : {best_mob['Val Accuracy (%)']}%")
print(f"  * Best Validation Loss  : {best_mob['Best Val Loss']}")

selection_manifest = {
    "custom_cnn_winner": {k: v for k, v in best_cnn.items() if k != "raw_macro_f1"},
    "custom_cnn_raw_macro_f1": best_cnn["raw_macro_f1"],
    "mobilenet_v2_winner": {k: v for k, v in best_mob.items() if k != "raw_macro_f1"},
    "mobilenet_v2_raw_macro_f1": best_mob["raw_macro_f1"],
    "selection_metric": "Validation Macro-F1 (Full Precision)",
    "validation_split_samples": 780,
    "train_split_samples": 3795,
    "held_out_test_evaluated": False,
    "all_5_runs_verified": True,
}

selection_path = RESULTS_ROOT / "candidate_selection.json"
selection_path.write_text(json.dumps(selection_manifest, indent=2))
print(f"\\nSaved candidate selection report to: {selection_path}")"""))

# Cell 10: Package All Screening Artifacts
cells.append(make_cell("code", """import shutil
from pathlib import Path

archive_target = Path("/kaggle/working/HAR-MMAct-screening-results")
print(f"[Step 5] Archiving screening outputs from {RESULTS_ROOT} to {archive_target}.zip...")

shutil.make_archive(str(archive_target), "zip", RESULTS_ROOT)
zip_size_mb = Path(f"{archive_target}.zip").stat().st_size / (1024**2)
print(f"Created archive: {archive_target}.zip ({zip_size_mb:.2f} MB)")
print("Screening workflow complete. Download archive from Kaggle Working directory.")"""))

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_nb = Path("notebooks/kaggle_screening_experiments.ipynb")
with open(out_nb, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Updated and verified Kaggle notebook: {out_nb}")
