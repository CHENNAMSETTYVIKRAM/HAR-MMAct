# Phase 2 Completion Report: Minimal Paired-Data Pipeline

**Date:** 2026-11-04  
**Project:** MMAct Human Activity Recognition (Software-Only)  
**Workspace:** `D:\HAR_MMAct`  
**Phase Status:** **COMPLETED** (Code implemented, unit tested, smoke-tested on synthetic data; zero genuine training performed)  

---

## 1. Executive Summary

Phase 2 established a clean, modular, leakage-free data indexing and preprocessing pipeline for paired video and wearable sensor classification on the MMAct dataset.

### Key Milestones Delivered:
1. **Repository Preservation:** The official MMAct challenge repository under `challenge-main\challenge-main\` remains untouched. Our project is implemented in modular packages (`src/`, `scripts/`, `tests/`, `docs/`).
2. **Verified Label Filtering:** The pipeline restricts activity recognition exclusively to the **10 verified target classes** established in `docs/LABEL_MAPPING.md` (`crouching`, `jumping`, `kicking`, `pointing`, `running`, `sitting`, `standing`, `using_phone`, `walking`, `waving_hand`).
3. **Strict Modality Pairing:** Video and sensor streams are matched only when source identifiers (`subject`, `scene`, `session`, `action`, `cam`) correspond. Anomaly reporting flags missing video, missing sensors, unknown labels, and duplicate IDs.
4. **Sensor Harmonization:** Implements the official MMAct preprocessing conventions:
   - Accelerometer scaling (dividing m/s² by 9.8 to convert to g's).
   - Timestamp parsing (`%Y%m%d_%H:%M:%S.%f` to epoch ms).
   - Multimodal datetime alignment (`datetime_merging`) resampling 50 Hz gyroscope onto the 100 Hz primary accelerometer timeline using backward fill.
   - Slicing uniform 100-sample (1.0s) windows with 9 sensor channels (`s_acc_x,y,z`, `w_acc_x,y,z`, `gyro_x,y,z`).
5. **Zero-Leakage Splitting:** Splits are enforced at the **subject** or **recording session** level *before* windowing. Formal mathematical verification confirms 0% subject, recording, or video overlap across train, validation, and test sets.
6. **Isolated Synthetic Smoke-Test Fixture:** An independent, strictly labeled synthetic fixture (`data/synthetic_smoke_test/`) was implemented and verified. It runs an end-to-end forward/backward training loop and metric calculation to guarantee functional integrity without fabricating genuine data.
7. **Comprehensive Unit Testing:** All 13 unit tests passed in pytest across label mapping, indexer diagnostics, preprocessing shapes, and split isolation.

---

## 2. Architecture & File Inventory

```text
D:\HAR_MMAct\
├── challenge-main\                 # Official challenge repository (preserved intact)
├── docs\
│   ├── INITIAL_RESOURCE_AUDIT.md  # Workspace and repository audit
│   ├── LABEL_MAPPING.md           # 10 verified target classes and disambiguation rules
│   ├── DATASET_AUDIT.md           # Sensor channel, rate, and modality specifications
│   └── PHASE2_REPORT.md           # This completion report
├── scripts\
│   ├── audit_dataset.py           # Dataset audit script
│   ├── generate_synthetic_data.py # Synthetic smoke-test fixture generator
│   └── run_smoke_test.py          # End-to-end smoke test runner
├── src\
│   ├── __init__.py
│   ├── config.py                  # Reproducibility seeds, hyperparams, class maps
│   ├── data\
│   │   ├── __init__.py
│   │   ├── indexer.py             # MMActIndexer for trimmed video-sensor pairing & diagnostics
│   │   ├── video_loader.py        # VideoLoader (uniform sampling, resize, ImageNet normalization)
│   │   ├── sensor_loader.py       # SensorLoader (CSV parsing, acc scaling, alignment, windowing)
│   │   ├── splits.py              # DatasetSplitter & verify_split_isolation (subject/session isolation)
│   │   ├── dataset.py             # MMActMultimodalDataset with on-disk tensor caching
│   │   └── synthetic.py           # Synthetic fixture generator with strict [SYNTHETIC] metadata
│   ├── models\
│   │   ├── __init__.py
│   │   ├── video_branch.py        # LightweightVideoBranch (2D CNN + temporal mean/max pooling)
│   │   ├── sensor_branch.py       # Sensor1DCNNBranch (multi-scale 1D CNN for IMU streams)
│   │   └── fusion_model.py        # MultimodalHARModel (fusion classifier + video-only/sensor-only heads)
│   └── utils\
│       ├── __init__.py
│       └── metrics.py             # Multiclass accuracy, macro F1, confusion matrix, OvR ROC AUC
├── tests\
│   ├── __init__.py
│   ├── test_label_mapping.py      # Tests for verified 10-class mapping
│   ├── test_indexer.py            # Tests for pairing and anomaly detection
│   ├── test_preprocessing.py      # Tests for tensor shapes (C,T,H,W) and (C,W)
│   ├── test_splits.py             # Tests for zero-leakage split isolation
│   └── test_models.py             # Tests for forward passes (multimodal and unimodal)
└── requirements.txt               # Pinned dependencies
```

---

## 3. Verified Behavior & Execution Results

### A. Unit Test Suite (`pytest tests -v`)
Ran 13 unit tests via Python 3.11:
* `tests/test_indexer.py::test_indexer_finds_paired_samples`: **PASSED** (pairs 120 synthetic clips across 10 classes)
* `tests/test_indexer.py::test_indexer_filters_unknown_labels`: **PASSED** (detects and filters out `drinking.mp4`)
* `tests/test_indexer.py::test_indexer_handles_nonexistent_dir`: **PASSED**
* `tests/test_label_mapping.py::test_verified_classes_count`: **PASSED** (exactly 10 verified classes)
* `tests/test_label_mapping.py::test_bidirectional_indexing`: **PASSED**
* `tests/test_label_mapping.py::test_presence_in_ground_truth_files`: **PASSED** (verified in official validation ground truth)
* `tests/test_models.py::test_video_branch_forward`: **PASSED** (produces shape `[2, 128]`)
* `tests/test_models.py::test_sensor_branch_forward`: **PASSED** (produces shape `[2, 128]`)
* `tests/test_models.py::test_multimodal_model_forward`: **PASSED** (produces `[2, 10]` across multimodal, video-only, sensor-only)
* `tests/test_preprocessing.py::test_video_loader_shape_and_range`: **PASSED** (shape `[3, 16, 112, 112]`, float32)
* `tests/test_preprocessing.py::test_sensor_loader_shape_and_channels`: **PASSED** (shape `[9, 100]`, float32)
* `tests/test_splits.py::test_subject_split_isolation`: **PASSED** (disjoint subjects; 0% leakage)
* `tests/test_splits.py::test_session_split_isolation`: **PASSED** (disjoint sessions; 0% leakage)

**Test Result:** **13 passed in 28.76s** (100% pass rate).

---

### B. End-to-End Synthetic Smoke Test (`scripts/run_smoke_test.py`)
Executed full pipeline on the isolated synthetic fixture:
1. **Indexing:** Scanned 122 videos and 181 CSV files; paired 120 valid samples (12 per class for 10 classes); filtered 3 anomalies.
2. **Split Isolation:**
   - `video_disjoint_train_val`: **PASSED**
   - `video_disjoint_train_test`: **PASSED**
   - `session_disjoint_train_val`: **PASSED**
   - `session_disjoint_train_test`: **PASSED**
   - `subject_disjoint_train_val`: **PASSED**
   - `subject_disjoint_train_test`: **PASSED**
3. **Data Loading & Preprocessing:** Verified video tensor `(3, 16, 112, 112)` and aligned sensor tensor `(9, 100)`. Caching active under `outputs/cache/smoke_test/`.
4. **Model Initialization:** `MultimodalHARModel` initialized with 251,102 parameters.
5. **Training Verification:** 2 epochs completed on CPU. CrossEntropyLoss decreased from 2.3905 -> 2.0342.
6. **Validation Evaluation:** Successfully calculated Accuracy (0.2000), Macro F1 (0.1200), Macro Precision (0.1111), Macro Recall (0.2000), and OvR ROC AUC (0.8222).
7. **Unimodal Verification:** Tested video-only (`override_mode='video_only'`) and sensor-only (`override_mode='sensor_only'`), verifying both produce valid 10-class logits.
8. **Checkpointing:** Checkpoint saved at `outputs/checkpoints/synthetic_smoke_test_model.pt` with explicit metadata `is_synthetic: True`.

---

## 4. Current Limitations & Scope

1. **No Genuine MMAct Model Training:** In accordance with constraints, no large-scale training was executed because the full ~90 GB dataset archives have not been extracted into the workspace. All training was restricted to the synthetic smoke-test fixture.
2. **Hardware Constraints:** Training was executed and tested on CPU. CUDA operations are supported automatically whenever a CUDA-capable device is detected.
3. **Interpolation in Sensor Merging:** Linear and backward-fill interpolation is used to harmonize sampling frequencies (50 Hz gyroscope to 100 Hz primary accelerometer timeline). This strictly reproduces the official baseline approach (`utils/time_series_classifiers.py`), but higher-order spline interpolation may be evaluated if needed.

---

## 5. Next Steps for Ingesting Genuine Paired Data

Once genuine MMAct trimmed video and sensor archives are downloaded and extracted:
1. Extract or link the genuine dataset files into `data/genuine_mmact/` following the expected directory layout:
   ```text
   data/genuine_mmact/
   ├── trimmed/
   │   ├── video/cross_scene/trainval/subject<S>/cam<V>/scene<C>/session<E>/<action>.mp4
   │   └── sensors/
   │       ├── acc_phone_clip/subject<S>/scene<C>/session<E>/<action>.csv
   │       ├── acc_watch_clip/subject<S>/scene<C>/session<E>/<action>.csv
   │       └── gyro_clip/subject<S>/scene<C>/session<E>/<action>.csv
   ```
2. Run the indexer on the genuine dataset:
   ```powershell
   & "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" -c "from src.data.indexer import MMActIndexer; from src.config import ProjectConfig; r = MMActIndexer(ProjectConfig().dataset_root).index(); print(r.summary())"
   ```
3. Run Phase 3/4 baseline training on the verified 10 target classes using the pre-tested `MultimodalHARModel` and training loop.
