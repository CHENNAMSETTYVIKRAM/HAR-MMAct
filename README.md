# HAR-MMAct: Multimodal Human Activity Recognition

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: Proprietary / Source--Available](https://img.shields.io/badge/License-Proprietary%20%2F%20Source--Available-red.svg)](LICENSE)
[![MMAct Benchmark](https://img.shields.io/badge/Benchmark-MMAct%20ICCV%202019-green.svg)](https://mmact19.github.io/challenge/)

A software-only, reproducible multimodal Human Activity Recognition (HAR) framework pairing video clips with synchronized 9-channel wearable inertial data from the [MMAct dataset](https://mmact19.github.io/challenge/).

---

## 1. Project Overview & Key Results

HAR-MMAct fuses spatial-temporal video representations (MobileNetV2 / Custom CNN) and 9-channel wearable inertial data (InceptionTime 1D CNN / Temporal Convolutional Networks) to classify 10 physical activity classes. Subject, recording, and session isolation checks passed under a custom subject-independent evaluation protocol.

### Sensor Modality Specifications:
The model consumes nine inertial sensor channels:
- **Phone accelerometer:** 3 channels ($x, y, z$).
- **Watch accelerometer:** 3 channels ($x, y, z$).
- **Gyroscope:** 3 channels ($x, y, z$).

> [!NOTE]
> An orientation sensor stream exists in the raw MMAct dataset, but it is **not** included in the nine-channel model input. The model does not use all four sensor streams; it operates strictly on the nine channels listed above.

### Benchmark Highlights:
- **Validation-Based Model Selection:** Checkpoint selection was conducted strictly using validation performance on the 780-sample validation split ($N=780$). The top-performing validation model was `Primary Multimodal` (`exp1_multimodal` — MobileNetV2 + InceptionTime Late Fusion), achieving **81.15% Validation Accuracy** and **0.8058 Macro F1**.
- **Locked Final Held-Out Test Evaluation:** All nine trained model configurations were evaluated on the identical 852-sample subject-independent held-out test split ($N=852$ samples, 132 recordings). Held-out test evaluation was performed after model training and checkpoint selection:
  - **Primary Multimodal Test Accuracy:** **76.7606%** (`exp1_multimodal`)
  - **Primary Multimodal Macro F1:** **0.7544618** (`exp1_multimodal`)
  - **Primary Multimodal Macro Precision:** **0.7843291** (`exp1_multimodal`)
  - **Primary Multimodal Macro Recall:** **0.7651515** (`exp1_multimodal`)
  - **Multiclass ROC-AUC (One-vs-Rest):** **0.9661350** (`exp1_multimodal`) / **0.9662302** (`multimodal_fusion`)
  - **Macro Average Precision:** **0.8589989** (`exp1_multimodal`)
- **Measured Accuracy Differences:**
  - Primary Multimodal improved test accuracy by approximately **2.11 percentage points** over the best sensor-only model (`tcn_sensor_only` at 74.6479%).
  - Primary Multimodal improved test accuracy by approximately **8.22 percentage points** over the retrained InceptionTime sensor-only baseline (`sensor_only_inception_time_v2_retrained` at 68.5446%).
  - Primary Multimodal improved test accuracy by approximately **34.27 percentage points** over the best video-only model (`mobilenet_v2_staged_v2_retrained` at 42.4883%).
  - *Note:* No statistical significance testing has been documented; these figures reflect measured empirical differences across the evaluated splits.
- **Documentation Reference:** Detailed methodology and analysis reports are documented in [`docs/FINAL_PROJECT_REPORT.md`](docs/FINAL_PROJECT_REPORT.md).

---

## 2. Comprehensive Nine-Model Performance Benchmark

All nine model configurations were evaluated on the held-out test set ($N=852$). Checkpoints were selected strictly using validation metrics.

The canonical benchmark values match [`outputs/final_test/all_models_rebuilt/nine_model_comparison.csv`](outputs/final_test/all_models_rebuilt/nine_model_comparison.csv) and [`outputs/final_test/all_models_rebuilt/nine_model_metrics.json`](outputs/final_test/all_models_rebuilt/nine_model_metrics.json) exactly:

| Model | Epochs Completed | Selected Epoch | Train Accuracy | Validation Accuracy | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | ROC-AUC OvR | Macro AP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Primary Multimodal | 50 | 49 | 98.21% | 81.15% | 76.7606% | 0.7544618 | 0.7843291 | 0.7651515 | 0.9661350 | 0.8589989 |
| Additional Multimodal Fusion | 50 | 30 | 96.79% | 80.38% | 75.4695% | 0.7464000 | 0.7770245 | 0.7537879 | 0.9662302 | 0.8499825 |
| Gated Multimodal TCN | 50 | 31 | 98.42% | 77.69% | 73.9437% | 0.7241405 | 0.7698761 | 0.7391775 | 0.9582143 | 0.8267652 |
| Sensor-only TCN | 50 | 42 | 96.71% | 77.44% | 74.6479% | 0.7190475 | 0.8133463 | 0.7428571 | 0.9523810 | 0.8093668 |
| Sensor-only InceptionTime | 50 | 42 | 84.43% | 76.92% | 68.5446% | 0.6562830 | 0.7069566 | 0.6809524 | 0.9311055 | 0.7741910 |
| Retrained MobileNetV2 | 44 | 32 | 55.96% | 36.54% | 42.4883% | 0.4316926 | 0.4665496 | 0.4551677 | 0.8442961 | 0.4731530 |
| Original MobileNetV2 | 50 | 41 | 49.99% | 35.26% | 42.1362% | 0.4153285 | 0.4473264 | 0.4589827 | 0.8608784 | 0.4982421 |
| Original Custom CNN | 50 | 47 | 15.97% | 14.74% | 15.3756% | 0.1139180 | 0.1540507 | 0.1726732 | 0.6160074 | 0.1791272 |
| Retrained Custom CNN V2 | 30 | 17 | 16.06% | 13.46% | 13.2629% | 0.0939734 | 0.1307080 | 0.1449134 | 0.6090665 | 0.1785648 |

### Measured Benchmark Findings:
1. **Top Metric Achievers:**
   - **Primary Multimodal** achieved the highest test accuracy (**76.76%**) and highest Macro F1 (**0.7545**).
   - **Additional Multimodal Fusion** achieved the highest ROC-AUC (**approximately 0.9662**).
   - **Sensor-only TCN** achieved the highest Macro Precision (**approximately 0.8133**).
2. **Modality Comparisons:**
   - Primary Multimodal improved test accuracy by approximately **2.11 percentage points** over Sensor-only TCN.
   - Primary Multimodal improved test accuracy by approximately **8.22 percentage points** over Sensor-only InceptionTime.
   - Primary Multimodal improved test accuracy by approximately **34.27 percentage points** over the best video-only model.
3. **Evaluation Metric Formulation:**
   - Multiclass ROC-AUC is calculated using One-vs-Rest (OvR) averaged macro-wise across classes.
   - Macro Average Precision is calculated from predicted class probabilities using one-hot encoded ground truth.
4. **Validation Isolation:**
   - Model checkpoints were selected based strictly on validation split performance. Test data was never utilized for checkpoint selection, loss computation, or hyperparameter optimization.

---

## 3. Verified Dataset and Split Details

The project processes paired video and 9-channel sensor recordings from the genuine MMAct dataset.

### Partition Breakdown:
- **Total Paired Samples:** 5,427 across 10 target classes.
- **Training Set:** 3,795 samples (595 recordings).
- **Validation Set:** 780 samples (124 recordings).
- **Held-Out Test Set:** 852 samples (132 recordings).
- **Total Recordings:** 851 recordings ($595 + 124 + 132$).
- **Subject Partitioning:**
  - **Training:** Subjects 1–14
  - **Validation:** Subjects 15–17
  - **Test:** Subjects 18–20

> [!IMPORTANT]
> **Custom Subject-Independent Split Disclaimer:**
> This project uses a custom, subject-independent evaluation split, **not** the official MMAct cross-scene or cross-view competition split. In the official MMAct competition, the test set was held private without public ground-truth labels. The custom subject-independent split partitions subjects strictly across splits, ensuring that subject, recording, and session isolation checks passed.

### Test Set Identity & CSV Row Ordering:
All nine model prediction files evaluate the exact same 852 sample IDs (`same_test_membership: True`, verified in [`outputs/final_test/all_models_rebuilt/final_output_audit.csv`](outputs/final_test/all_models_rebuilt/final_output_audit.csv)). The row ordering across individual prediction CSV files can differ due to multi-threaded PyTorch DataLoader worker scheduling; however, test-set membership is 100% identical.

---

## 4. Dataset Downloads & Directory Layout

Dataset archives are hosted on Kaggle and must be downloaded prior to training:

### Kaggle Datasets:
1. **[Trimmed Scene Video Dataset](https://www.kaggle.com/datasets/vikramche/trimmed-scene-video)**: Trimmed MP4 video clips organized by subject, camera view, scene, and session.
2. **[MMAct Trimmed Dataset](https://www.kaggle.com/datasets/vikramche/mmact-trimmed)**: Four wearable inertial sensor streams (`acc_phone_clip`, `acc_watch_clip`, `gyro_clip`, `orientation_clip`), official split files, and baseline trimmed clips.
   - *Note:* The model uses `acc_phone`, `acc_watch`, and `gyro` (3 channels each = 9 channels total). `orientation_clip` is present in the archive but not included in model inputs.

### Expected Directory Layout:
Extract files into `data/genuine_mmact/`:
```text
data/genuine_mmact/
└── trimmed/
    ├── video/
    │   └── cross_scene/
    │       └── trainval/
    │           └── subject<ID>/
    │               └── cam<ID>/
    │                   └── scene<ID>/
    │                       └── session<ID>/
    │                           └── <action>.mp4
    ├── sensors/
    │   ├── acc_phone_clip/subject<ID>/scene<ID>/session<ID>/<action>.csv
    │   ├── acc_watch_clip/subject<ID>/scene<ID>/session<ID>/<action>.csv
    │   ├── gyro_clip/subject<ID>/scene<ID>/session<ID>/<action>.csv
    │   └── orientation_clip/subject<ID>/scene<ID>/session<ID>/<action>.csv
    └── splits/
        ├── trimmed_x-scene_train.txt
        ├── trimmed_x-scene_val.txt
        ├── trimmed_x-view_train.txt
        └── trimmed_x-view_val.txt
```

### Verification Command:
```powershell
python scripts/audit_dataset.py
```

---

## 5. 10 Verified Target Activity Classes

The 10 project activity classes verified against ground truth files and [`docs/LABEL_MAPPING.md`](docs/LABEL_MAPPING.md):

| Class Index | Target Class | Physical Activity Description |
|:---:|---|---|
| `0` | `crouching` | Lowering torso and bending knees into a crouched position |
| `1` | `jumping` | Upward vertical propulsion followed by landing |
| `2` | `kicking` | Unilateral forward leg extension |
| `3` | `pointing` | Directional arm extension and holding |
| `4` | `running` | Continuous high-cadence locomotor gait |
| `5` | `sitting` | Transitioning to or resting in a seated posture |
| `6` | `standing` | Upright stationary vertical posture |
| `7` | `using_phone` | Holding and manipulating a smartphone near chest/face level |
| `8` | `walking` | Standard forward walking locomotion |
| `9` | `waving_hand` | Oscillating wrist movement above waist level |

---

## 6. Model Architectures

The framework implements multiple model families across modalities:

```
 [Video: (B, 3, 16, 112, 112)]        [Sensors: (B, 9, 100)]
               │                                │
     MobileNetV2 (Pretrained)         InceptionTime 1D CNN
               │                                │
      Dual Temporal Pooling             Global Average Pool
               │                                │
      Video Vector (128-d)             Sensor Vector (128-d)
               └───────────────┬────────────────┘
                               │
                       Late Concatenation
                            (256-d)
                               │
                         MLP Head
              (Linear 128 -> LayerNorm -> GELU)
                               │
                       Logits (10 classes)
```

### Implemented Model Families:

1. **Video Branch Architectures:**
   - **`MobileNetV2VideoBranch`:** ImageNet-pretrained inverted residual blocks extracting 1280-dim per-frame representations from $T=16$ uniform frames ($112 \times 112$). Applies temporal Mean + Max pooling ($1280 \times 2 = 2560\text{-d}$) followed by a Linear projection with BatchNorm1d and ReLU to produce a 128-dim video embedding vector.
   - **`CustomCNNVideoBranch`:** A 3-stage custom 2D convolutional baseline trained from scratch with Batch Normalization, SpatialDropout2d, per-frame adaptive pooling to 128 dims, and temporal mean+max pooling projected to 128 dims.
2. **Sensor Branch Architectures:**
   - **`InceptionTimeSensorBranch`:** 1D InceptionTime architecture (Fawaz et al., 2020) with multi-scale 1D convolutions (kernel sizes 9, 19, 39), bottleneck projection (32 filters), and residual connections across 9 inertial channels. Operates on 100-sample windows (1.0 s at 100 Hz), pooled via AdaptiveAvgPool1d and projected with BatchNorm1d and ReLU to a 128-dim sensor embedding vector.
   - **`TemporalConvSensorBranch` (TCN):** 1D Temporal Convolutional Network baseline with dilated convolutions capturing temporal receptive fields across the 9 sensor channels.
3. **Fusion Modules:**
   - **Late Concatenation (`MultimodalHARModel`):** Concatenates 128-dim video and 128-dim sensor embeddings into a 256-dim joint vector, followed by an MLP classifier: `Linear(256 -> 128)` $\rightarrow$ `LayerNorm(128)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(128 -> 10)`. Auxiliary heads allow isolated video-only and sensor-only inference.
   - **Gated Fusion:** Employs a learnable gating network (`Linear(256 -> 128)` $\rightarrow$ `GELU` $\rightarrow$ `Linear(128 -> 2)`) to dynamically weight visual and inertial representations before classification.

---

## 7. Verified Model Checkpoints (All 9 Models)

All nine trained best checkpoint binaries are physically present in this repository:

| # | Experiment Name | Checkpoint Path | Architecture | Checkpoint Size |
|:---:|---|---|---|:---:|
| 1 | **Primary Multimodal** | [`outputs/exp1_multimodal/checkpoints/best_model.pth`](outputs/exp1_multimodal/checkpoints/best_model.pth) | MobileNetV2 + InceptionTime | 32.94 MB |
| 2 | **Additional Multimodal Fusion** | [`outputs/multimodal_fusion/checkpoints/best_model_multimodal.pth`](outputs/multimodal_fusion/checkpoints/best_model_multimodal.pth) | MobileNetV2 + InceptionTime | 32.95 MB |
| 3 | **Gated Multimodal TCN** | [`outputs/tcn_gated_fusion/checkpoints/best_model_tcn_gated_fusion.pth`](outputs/tcn_gated_fusion/checkpoints/best_model_tcn_gated_fusion.pth) | MobileNetV2 + TCN 1D | 32.96 MB |
| 4 | **Sensor-only TCN** | [`outputs/tcn_sensor_only/checkpoints/best_model_tcn_sensor.pth`](outputs/tcn_sensor_only/checkpoints/best_model_tcn_sensor.pth) | TCN 1D | 3.36 MB |
| 5 | **Sensor-only InceptionTime** | [`outputs/sensor_only_inception_time_v2_retrained/checkpoints/best_model_cnn.pth`](outputs/sensor_only_inception_time_v2_retrained/checkpoints/best_model_cnn.pth) | InceptionTime 1D | 3.73 MB |
| 6 | **Retrained MobileNetV2** | [`outputs/mobilenet_v2_staged_v2_retrained/checkpoints/best_model_mobilenet_v2.pth`](outputs/mobilenet_v2_staged_v2_retrained/checkpoints/best_model_mobilenet_v2.pth) | MobileNetV2 | 30.96 MB |
| 7 | **Original MobileNetV2** | [`outputs/mobilenet_v2/checkpoints/best_model_mobilenet_v2.pth`](outputs/mobilenet_v2/checkpoints/best_model_mobilenet_v2.pth) | MobileNetV2 | 30.96 MB |
| 8 | **Retrained Custom CNN V2** | [`outputs/improved_cnn_v2_retrained_v2/checkpoints/best_model_custom_cnn.pth`](outputs/improved_cnn_v2_retrained_v2/checkpoints/best_model_custom_cnn.pth) | Custom CNN 2D | 2.84 MB |
| 9 | **Original Custom CNN** | [`outputs/improved_cnn/checkpoints/best_model_cnn.pth`](outputs/improved_cnn/checkpoints/best_model_cnn.pth) | Custom CNN 2D | 2.84 MB |

---

## 8. Consolidated Comparison Plots & Visualizations

Consolidated benchmark artifacts are located under [`outputs/final_test/all_models_rebuilt/`](outputs/final_test/all_models_rebuilt/):

- **[Training vs Testing Accuracy](outputs/final_test/all_models_rebuilt/training_vs_testing_accuracy.png):** Compares training accuracy at the selected checkpoint epoch against final held-out test accuracy across all nine models.
- **[Training Accuracy Comparison](outputs/final_test/all_models_rebuilt/training_accuracy_comparison.png):** Comparison of model convergence and checkpoint training accuracy.
- **[Testing Accuracy Comparison](outputs/final_test/all_models_rebuilt/testing_accuracy_comparison.png):** Test accuracy bar chart comparing all nine configurations on the 852 held-out samples.
- **[Macro F1 Comparison](outputs/final_test/all_models_rebuilt/macro_f1_comparison.png):** Macro F1 score comparison across models.

### Experiment Evaluation Artifacts:
Individual experiment directories contain:
- Training and validation curves (`outputs/<exp>/history.json`, `training_validation_accuracy.png`, `training_validation_loss.png`)
- Evaluation metrics JSON and prediction CSV (`final_test_evaluation.json`, `final_test_predictions.csv`)
- Confusion matrices (`test_confusion_matrix.png`)
- Multiclass ROC curves (`test_roc_ovr.png`)
- Precision-recall curves (`test_pr_curves.png`)

---

## 9. Top Multimodal Per-Class Test Performance (Measured Results)

Class-level test metrics for `Primary Multimodal` (`exp1_multimodal`) on the 852 held-out test samples ([`outputs/exp1_multimodal/test_evaluation_v2.json`](outputs/exp1_multimodal/test_evaluation_v2.json)):

| Activity Class | Test Support | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| `crouching` | 84 | 0.6754 | 0.9167 | 0.7778 |
| `jumping` | 84 | 0.3492 | 0.2619 | 0.2993 |
| `kicking` | 84 | 0.6351 | 0.5595 | 0.5949 |
| `pointing` | 84 | 1.0000 | 0.6786 | 0.8085 |
| `running` | 84 | 1.0000 | 0.4524 | 0.6230 |
| `sitting` | 48 | 0.9600 | 1.0000 | 0.9796 |
| `standing` | 84 | 0.6357 | 0.9762 | 0.7700 |
| `using_phone` | 132 | 0.8725 | 0.9848 | 0.9253 |
| `walking` | 84 | 0.8023 | 0.8214 | 0.8118 |
| `waving_hand` | 84 | 0.9130 | 1.0000 | 0.9545 |

### Measured Observations:
- **High-F1 Classes:** `sitting` achieved 0.9796 F1 (1.0000 recall on 48 samples), `waving_hand` achieved 0.9545 F1 (1.0000 recall on 84 samples), and `using_phone` achieved 0.9253 F1 on 132 samples.
- **Flawless Precision:** Both `pointing` and `running` achieved 1.0000 precision (zero false positives recorded), though `running` exhibited lower recall (0.4524).
- **Challenging Class:** `jumping` was the lowest-performing class, recording 0.2993 F1 (0.3492 precision, 0.2619 recall) on the held-out test split.

---

## 10. Environment Setup & Execution

### Setup Virtual Environment:
```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Run Verification Test Suite:
```powershell
# Run data alignment regression tests
python -m pytest tests/test_accuracy_fixes.py -v

# Run model architecture and split isolation tests
python -m pytest tests/test_models.py tests/test_splits.py -v
```

---

## 11. Standalone Inference CLI

The inference CLI ([`scripts/infer.py`](scripts/infer.py)) supports three prediction modes based on provided input flags:
1. **Multimodal Fused Inference:** Provided with `--video` and sensor flags (`--sensor_phone`, `--sensor_watch`, `--sensor_gyro`).
2. **Video Stream Alone:** Provided with `--video` alone.
3. **Sensor Stream Alone:** Provided with sensor flags alone.

### A. Multimodal Inference (Video + Sensors):
```powershell
python scripts/infer.py `
    --video challenge-main/challenge-main/utils/sample_clip/carrying.mp4 `
    --sensor_phone path/to/acc_phone.csv `
    --sensor_watch path/to/acc_watch.csv `
    --sensor_gyro path/to/gyro.csv `
    --checkpoint outputs/exp1_multimodal/checkpoints/best_model.pth `
    --top_k 3
```

### B. Video Stream Alone:
```powershell
python scripts/infer.py `
    --video challenge-main/challenge-main/utils/sample_clip/carrying.mp4 `
    --checkpoint outputs/exp1_multimodal/checkpoints/best_model.pth
```

### C. Sensor Stream Alone:
```powershell
python scripts/infer.py `
    --sensor_phone path/to/acc_phone.csv `
    --sensor_watch path/to/acc_watch.csv `
    --sensor_gyro path/to/gyro.csv `
    --checkpoint outputs/exp1_multimodal/checkpoints/best_model.pth
```

### Verified Demo Output (Video Stream Alone):
```text
============================================================
 HAR_MMAct: Inference Prediction
============================================================
[Mode: Video Stream Alone (Multimodal Architecture)]

Top-3 Activity Predictions:
  1. waving_hand     :  11.75%
  2. sitting         :  11.61%
  3. jumping         :  10.94%
============================================================
```

> [!NOTE]
> **Inference Demo Scope:** The demo execution recorded above ran `scripts/infer.py` on `challenge-main/challenge-main/utils/sample_clip/carrying.mp4` using the multimodal checkpoint (`outputs/exp1_multimodal/checkpoints/best_model.pth`). Because sensor files were not provided in that execution, the CLI operated strictly in **video-stream-only mode** via the auxiliary video head. An end-to-end multimodal demo requiring concurrent video and sensor files was not executed in that particular test.

---

## 12. Kaggle Cloud GPU Execution

For execution on Kaggle GPU:
1. Open [`notebooks/ML_Trained_book/mmact-har-ml-model_Trained.ipynb`](notebooks/ML_Trained_book/mmact-har-ml-model_Trained.ipynb) or generate a fresh notebook with `python scripts/create_kaggle_notebook.py`.
2. Attach both Kaggle datasets:
   - `vikramche/trimmed-scene-video`
   - `vikramche/mmact-trimmed`
3. Enable GPU accelerator (`Tesla T4 x2` or `P100`) and set Internet to `On`.
4. Run all cells top-to-bottom. The automated staging discovers attached inputs, verifies integrity, caches preprocessed tensors, trains benchmark models, and exports result archives.

---

## 13. Copyright, Ownership & Licensing

- **Original Project Creator & Copyright Holder:** **CHENNAMSETTY VIKRAM** ([@CHENNAMSETTYVIKRAM](https://github.com/CHENNAMSETTYVIKRAM))
- **Repository URL:** [`https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct`](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct)
- **Copyright Notice:** `Copyright (c) 2026 CHENNAMSETTY VIKRAM. All Rights Reserved.`
- **License Status:** **Proprietary & Source-Available** (Governed by [`LICENSE`](LICENSE)).
- **Inspection & Evaluation Rights:** Public inspection, code review, and non-commercial academic evaluation are welcomed and permitted.
- **Commercial Restrictions:** No commercial exploitation, sublicensing, SaaS deployment, or unauthorized resale is permitted without an express, separate written commercial license.
- **Collaboration & Contributions:** Access to this repository does not transfer or imply transfer of original intellectual property. Collaborators and contributors are subject to [`COLLABORATION_POLICY.md`](COLLABORATION_POLICY.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), and signed written agreements.
- **Third-Party Rights & Challenge-Main Disclaimer:** The folder `challenge-main/` contains the unmodified starter toolkit, evaluation scripts, and challenge materials from the **MMAct Challenge 2021** (Quan Kong, Norimichi Ukita, et al.; licensed under CC BY-NC 4.0). CHENNAMSETTY VIKRAM did not author or develop `challenge-main/`; it was added strictly as an external reference and benchmark evaluation toolkit. The Creator asserts copyright solely over original source code (`src/`), execution pipelines (`scripts/`), test suites (`tests/`), notebooks (`notebooks/`), and original documentation.
- **Legal References:** See full legal documentation in [`LICENSE`](LICENSE), [`COPYRIGHT.md`](COPYRIGHT.md), and [`IP_OWNERSHIP.md`](IP_OWNERSHIP.md).
