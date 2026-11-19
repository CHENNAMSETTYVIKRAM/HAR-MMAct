# HAR-MMAct — Fast Accuracy Improvement Report & Screening Audit

## 1. Initial Repository State and Branch

- **Repository**: `CHENNAMSETTYVIKRAM/HAR-MMAct`
- **Initial Branch**: `master` (ahead of `origin/master` by 1 local commit; uncommitted attribution/documentation files preserved untouched).
- **Target Branch**: `accuracy-improvements`
- **Synchronization**: `git fetch origin` followed by `git switch --track origin/accuracy-improvements` and `git pull --ff-only origin accuracy-improvements`.
- **Initial Commit on Branch**: `810cb2f Document fast accuracy findings and safeguards`
- **Working Tree Integrity**: All original checkpoints (`outputs/mobilenet_v2`, `outputs/mobilenet_v2_staged_v2_retrained`, `outputs/improved_cnn`, `outputs/improved_cnn_v2_retrained_v2`), manifests, metrics, histories, and plots were strictly preserved.

---

## 2. Confirmed Technical Defects

### Defect 1: Excessive Custom CNN Regularization & Overfitting Inability
- **Code Defect**: `MultimodalHARModel` in `src/models/fusion_model.py` clamped `dropout=max(0.35, dropout)` for `custom_cnn`, preventing users from selecting lower dropout rates such as `0.20` or `0.25`.
- **Branch Defect**: `CustomCNNVideoBranch` in `src/models/video_branch.py` had hardcoded `Dropout2d(0.15)` and `Dropout2d(0.20)` across frames. In addition, `video_only_classifier` applied two additional `Dropout(dropout)` stages.
- **Diagnostic Proof**:
  - With hardcoded spatial dropout (0.15 / 0.20) and high projection dropout (0.35), a 10-sample synthetic subset diagnostic failed completely: loss remained stuck at **2.2185** and accuracy at **10.0%** (random guessing).
  - When spatial dropout was reduced to `0.00` - `0.05`, the exact same architecture achieved **100.0%** accuracy and **0.0307** loss.
  - This confirmed why the historical Retrained Custom CNN V2 (13.26% test acc) performed *worse* than the Original Custom CNN (15.38% test acc).

### Defect 2: Unused Sensor and Fusion Parameters Entering Video-Only Optimizer
- **Code Defect**: In `scripts/train.py`, `MultimodalHARModel` was instantiated in `video_only` mode. Although `forward()` only executes `video_branch` and `video_only_classifier`, the model also creates `sensor_branch`, `fusion_classifier`, and `sensor_only_classifier`.
- **Impact**: Total model parameters were 438,910, but active video-only parameters were only 144,682. The remaining **294,228 parameters (67%) were inactive**, yet were passed to AdamW:
  ```python
  optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], ...)
  ```
  This created useless optimizer momentum and variance states, and in staged fine-tuning, mixed inactive parameters into the head parameter group.

### Defect 3: Missing Audit and Initialization Tracking
- **Code Defect**: `MobileNetV2VideoBranch` did not record `normalization_mean`, `normalization_std`, or `pretrained_source` attributes, preventing automated test verification of transfer learning weights.
- **Manifest Defect**: `experiment_manifest.json` did not record `freeze_video`, `staged_finetune`, `unfreeze_epoch`, `unfreeze_lr`, `pretrained_source`, or `active_trainable_params`.

### Defect 4: Four-Sample Dataset Split Investigation (3,795 vs 3,799 Samples)
- **Investigation**: Manifest comparison between original (`outputs/mobilenet_v2/experiment_manifest.json`) and retrained (`outputs/mobilenet_v2_staged_v2_retrained/experiment_manifest.json`):
  - Original Train: 3,795 samples, 595 recordings, standing: 355.
  - Retrained Train: 3,799 samples, 599 recordings, standing: 359 (+4 recordings, all standing).
  - Validation: Exactly 780 samples, 124 recordings in both.
  - Test: Exactly 852 samples, 132 recordings in both.
  - Split Isolation: All checks (`subject_disjoint`, `recording_disjoint`, `session_disjoint`, `video_disjoint`) passed with zero overlap across train, val, and test.
- **Conclusion**: The 4 additional samples belong strictly to training subjects 1–14 and represent extra indexed training recordings in that environment. No data leakage or partition contamination exists.

---

## 3. Changes Actually Implemented

1. **`src/models/fusion_model.py`**:
   - Removed `dropout=max(0.35, dropout)` override for `custom_cnn`; passed `dropout=dropout` directly.
   - Preserved all existing `forward()` control flow, multimodal `else` branches, and sensor-only paths.

2. **`src/models/video_branch.py`**:
   - Added `self.normalization_mean = (0.485, 0.456, 0.406)` and `self.normalization_std = (0.229, 0.224, 0.225)`.
   - Added auditable `self.pretrained_source` reporting `"torchvision-default"`, `"torchvision-legacy"`, `"torchvision-random"`, `"pure-pytorch-external"`, or `"pure-pytorch-random"`.
   - Made `CustomCNNVideoBranch` spatial dropout configurable via `spatial_dropout1` and `spatial_dropout2` parameters while preserving existing state dict parameter names and checkpoint compatibility.

3. **`scripts/train.py`**:
   - Scoped active optimizer parameters based on mode: in `video_only`, only active video branch and classifier parameters enter AdamW; inactive modalities are excluded.
   - Scoped staged fine-tuning head parameters to active video parameters only.
   - Added audit fields (`freeze_video`, `staged_finetune`, `unfreeze_epoch`, `unfreeze_lr`, `pretrained_source`, `active_trainable_params`, `total_params`) to `experiment_manifest.json`.

4. **`tests/test_fast_accuracy_diagnostics.py`**:
   - Corrected gradient assertion to inspect `model.video_branch.named_parameters()` end-to-end.
   - Set `frozen.eval()` for single-sample inference check to adhere to `BatchNorm1d` contracts.

---

## 4. Screening Experiments Audit & Integrity Verification

An audit was conducted across all 5 local screening runs located under `outputs/diagnostics/`:

| Attribute | `cnn_dropout035` | `cnn_dropout020` | `cnn_dropout025_cw` | `mobilenet_frozen10` | `mobilenet_staged8` |
|---|---|---|---|---|---|
| **Execution Host** | Local (Windows, CPU) | Local (Windows, CPU) | Local (Windows, CPU) | Local (Windows, CPU) | Local (Windows, CPU) |
| **Dataset Source** | `data/synthetic_smoke_test` | `data/synthetic_smoke_test` | `data/synthetic_smoke_test` | `data/synthetic_smoke_test` | `data/synthetic_smoke_test` |
| **Synthetic Flag (`is_synthetic`)** | `True` | `True` | `True` | `True` | `True` |
| **Train Samples Count** | 40 | 40 | 40 | 40 | 40 |
| **Val Samples Count** | 40 | 40 | 40 | 40 | 40 |
| **Test Samples Count** | 40 (Not evaluated) | 40 (Not evaluated) | 40 (Not evaluated) | 40 (Not evaluated) | 40 (Not evaluated) |
| **Split Subjects** | Train: `subject1`, Val: `subject2`, Test: `subject3` | Train: `subject1`, Val: `subject2`, Test: `subject3` | Train: `subject1`, Val: `subject2`, Test: `subject3` | Train: `subject1`, Val: `subject2`, Test: `subject3` | Train: `subject1`, Val: `subject2`, Test: `subject3` |
| **Train Class Dist.** | 4 samples / class (10 classes) | 4 samples / class (10 classes) | 4 samples / class (10 classes) | 4 samples / class (10 classes) | 4 samples / class (10 classes) |
| **Val Class Dist.** | 4 samples / class (10 classes) | 4 samples / class (10 classes) | 4 samples / class (10 classes) | 4 samples / class (10 classes) | 4 samples / class (10 classes) |
| **Epochs Completed** | 10 | 10 | 10 | 10 | 8 (Server restart interrupted final save) |
| **Best Checkpoint Epoch** | Epoch 8 | Epoch 5 | Epoch 8 | Epoch 10 | Epoch 8 |
| **Checkpoint Selection Metric** | Validation Macro-F1 | Validation Macro-F1 | Validation Macro-F1 | Validation Macro-F1 | Validation Macro-F1 |
| **Random Seed** | 42 | 42 | 42 | 42 | 42 |
| **Complete 780-Val Split Evaluated?** | **NO** (40-sample fixture) | **NO** (40-sample fixture) | **NO** (40-sample fixture) | **NO** (40-sample fixture) | **NO** (40-sample fixture) |
| **Per-Sample `val_preds.csv` Saved?** | No (only summary/ckpt) | No (only summary/ckpt) | No (only summary/ckpt) | No (only summary/ckpt) | No (only ckpt) |

---

## 5. Independent Validation Metrics Recalculation

Using `scripts/verify_screening_metrics.py`, each saved checkpoint was loaded and independently re-evaluated on the exact validation split used during the screening runs:

```
Train/Val Overlap: 0 samples
Validation 10-Class Representation: 100% (4 samples per verified class)
```

| Checkpoint Name | Checkpoint File Path | Checkpoint Saved Metrics | Independently Recalculated Metrics | Discrepancy (Diff) |
|---|---|---|---|:---:|
| `cnn_dropout035` | `outputs/diagnostics/cnn_dropout035/checkpoints/best_model_custom_cnn.pth` | Loss: 2.0092<br>Acc: 60.00%<br>Macro-F1: 0.5067 | Loss: 2.0092<br>Acc: 60.00%<br>Macro-F1: 0.5067 | **0.000000** |
| `cnn_dropout020` | `outputs/diagnostics/cnn_dropout020/checkpoints/best_model_custom_cnn.pth` | Loss: 1.8954<br>Acc: 50.00%<br>Macro-F1: 0.4067 | Loss: 1.8954<br>Acc: 50.00%<br>Macro-F1: 0.4067 | **0.000000** |
| `cnn_dropout025_cw` | `outputs/diagnostics/cnn_dropout025_cw/checkpoints/best_model_custom_cnn.pth` | Loss: 1.6873<br>Acc: 60.00%<br>Macro-F1: 0.5500 | Loss: 1.6873<br>Acc: 60.00%<br>Macro-F1: 0.5500 | **0.000000** |
| `mobilenet_frozen10` | `outputs/diagnostics/mobilenet_frozen10/checkpoints/best_model_mobilenet_v2.pth` | Loss: 2.2673<br>Acc: 30.00%<br>Macro-F1: 0.1952 | Loss: 2.2673<br>Acc: 30.00%<br>Macro-F1: 0.1952 | **0.000000** |
| `mobilenet_staged8` | `outputs/diagnostics/mobilenet_staged8/checkpoints/best_model_mobilenet_v2.pth` | Loss: 2.3060<br>Acc: 10.00%<br>Macro-F1: 0.0400 | Loss: 2.3060<br>Acc: 10.00%<br>Macro-F1: 0.0400 | **0.000000** |

All recalculated values matched checkpoint headers to 6 decimal places.

---

## 6. Configuration Selection Decision & Benchmark Fair Comparison

### Custom CNN
- **Finding**: Trial 3 (`outputs/diagnostics/cnn_dropout025_cw`) achieved the highest Macro-F1 (0.5500) and lowest validation loss (1.6873) on the synthetic diagnostic split.
- **Selection Decision**: **Retained strictly as a Candidate Hypothesis**, not an established benchmark improvement. Because the screening ran on the 40-sample synthetic fixture, Trial 3 cannot be declared superior to the historical 15.38% benchmark until verified against the genuine 780-sample validation split.

### MobileNetV2
- **Finding**: The frozen configuration (`outputs/diagnostics/mobilenet_frozen10`) achieved 0.1952 Macro-F1 and 30.00% accuracy, outperforming staged fine-tuning on the synthetic diagnostic fixture.
- **Selection Decision**: **Explicitly NOT declared an improvement over the historical benchmark.** The historical benchmark achieved 35.26% validation accuracy and 42.14% test accuracy on the complete 780-sample validation and 852-sample test splits. Comparing a 40-sample synthetic screening result against the 780-sample benchmark is invalid. Both frozen and staged configurations must be screened fairly on the genuine 780-sample validation split before declaring a winner.

---

## 7. Kaggle Execution Preparation & Notebook Audit

To reproduce and evaluate the screening configurations on the complete genuine MMAct dataset (3,795 train samples, 780 validation samples), a Kaggle notebook was created, audited, and syntax-verified:

**Notebook File**: `notebooks/kaggle_screening_experiments.ipynb`  
**Generator Script**: `scripts/generate_kaggle_screening_notebook.py`

### Pre-Execution Integrity Audit of the Screening Notebook
1. **No Synthetic Contamination**: Dataset discovery in Cell 3 explicitly scans `/kaggle/input` using `discover_kaggle_input_roots` and `FastMMActStager`. Any path containing `"synthetic"` is strictly barred by assertion.
2. **Actual MMAct Directory Structure Supported**: Handles nested `/kaggle/input/datasets/<user>/<slug>` and `/kaggle/input/<slug>` structures, resolves duplicate recordings across split archives (e.g. subjects 8 and 16), and stages video, sensor, and official split files into canonical structure.
3. **Subject IDs and Recording Identifiers Preserved**: Canonical staging preserves the exact `(subject, cam, scene, session)` tuple keys without name collisions or loss of recording identity.
4. **Verified 10-Class Label Mapping**: Maps exclusively to `VERIFIED_TARGET_CLASSES` in `src/config.py`. Cell 4 asserts that all 10 verified action classes are present in the validation split.
5. **Absolute Test Isolation (Subjects 18–20 Never Accessed)**: Screening strictly uses train subjects 1–14 and validation subjects 15–17 only. Test subjects 18–20 are **never** constructed, iterated over, loaded, evaluated, or accessed during screening. The test split is explicitly omitted from construction (`test_samples = []`), and all screening trials pass `--no_test_split`. The held-out test protocol is preserved strictly for post-candidate final evaluation.
6. **Benchmark Split Consistency**: Train (3,795 samples) and validation (780 samples) counts, subject boundaries (train $\in$ [1..14], val $\in$ [15..17]), and all 10 verified target classes are verified against the original benchmark manifest (`outputs/mobilenet_v2/experiment_manifest.json`). If counts or identities differ, the screening stops immediately.
7. **MobileNetV2 Verified Pretrained Loading**: Cell 6 executes a preflight verification gate asserting `test_branch.pretrained_loaded == True` and `test_branch.pretrained_source in ["torchvision-default", "torchvision-legacy", "pure-pytorch-external"]`. If pretrained weights fail to load, execution aborts immediately; silent fallback to random initialization is barred.
8. **Pure Video-Only Isolation**: In `video_only` mode, the model receives only video tensors (`video=video, sensor=None`). Sensor tensors are never moved to GPU or passed to the model. In addition, `MMActIndexer` runs with `required_sensor_modalities=[]` in video-only mode so missing sensor modalities never block video indexing.
9. **Experiment Integrity & Rejection of Missing Metrics**: All five experiment summaries, manifests, and histories are required before candidate selection. Dependency installation failures are fatal (`check=True`). Metric fields (`macro_f1`, `accuracy`, `loss`) are required without zero-substitution. Exactly 12 completed epochs are enforced.
10. **Correct Best Epoch Derivation**: `best_epoch` is derived directly from maximum validation Macro-F1 or the training script's recorded `best_epoch` (not minimum validation loss). Missing or inconsistent history is strictly rejected.
11. **AST Code Syntax Verification**: All 9 code cells verified via `ast.parse` with 100% valid Python syntax.

---

## 8. Automated Test Suite Results

Full regression and diagnostics suite executed against active codebase:

```bash
.venv\Scripts\python.exe -m pytest -q tests/test_models.py tests/test_label_mapping.py tests/test_splits.py tests/test_fast_accuracy_diagnostics.py
```

**Result**: **37 passed in 30.76s (100% pass rate)**.
- `tests/test_fast_accuracy_diagnostics.py`: 9 passed (Custom CNN shapes, gradients, trainable parameter isolation, overfit diagnostic, MobileNet normalization, freeze/optimizer state, pre-indexing subject exclusion, manifest serialization round-trip, verified pretrained weights gate).
- `tests/test_models.py`: 12 passed (All backbone configurations, forward passes, temporal aggregation).
- `tests/test_label_mapping.py`: 7 passed (Action label indexing, canonical mappings, verification gates).
- `tests/test_splits.py`: 9 passed (Subject isolation, recording disjointness, zero test leakage).

---

## 9. Missing Evidence & Unresolved Issues

1. **Local Dataset Limitation**: The local repository workspace contains only subjects 8 and 16 under `data/genuine_mmact` (496 clips total: 284 train, 212 val, 0 test). Subjects 1–7 and 9–15 are absent locally; full 3,795-sample train and 780-sample validation screening requires Kaggle with the attached MMAct dataset.
2. **Absence of per-sample prediction CSVs in training runs**: `scripts/train.py` exports `training_summary.json` and checkpoints, but not per-sample `validation_predictions.csv` rows during training. The independent verification script `scripts/verify_screening_metrics.py` was created to bridge this gap.
3. **Execution Interruption on Staged Run**: `mobilenet_staged8` finished all 8 epochs but the manifest and summary were interrupted before disk flush due to host restart; checkpoint weights were safely preserved and independently verified.

---

## 10. Recommended Next Experiments

Execute `notebooks/kaggle_screening_experiments.ipynb` on Kaggle GPU across the complete 780-sample validation split:

```bash
# Custom CNN Screening Candidate
python scripts/train.py \
    --mode video_only \
    --video_backbone custom_cnn \
    --epochs 15 \
    --batch_size 16 \
    --lr 1e-3 \
    --dropout 0.25 \
    --use_class_weights \
    --benchmark_warmup \
    --output_dir outputs/screening/cnn_dropout025_cw

# MobileNetV2 Frozen Screening Baseline
python scripts/train.py \
    --mode video_only \
    --video_backbone mobilenet_v2 \
    --epochs 15 \
    --batch_size 16 \
    --lr 1e-4 \
    --dropout 0.20 \
    --freeze_video \
    --use_class_weights \
    --benchmark_warmup \
    --output_dir outputs/screening/mobilenet_frozen15
```
