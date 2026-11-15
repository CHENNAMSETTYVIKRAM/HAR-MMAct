# HAR-MMAct: Multimodal Human Activity Recognition

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MMAct Benchmark](https://img.shields.io/badge/Benchmark-MMAct%20ICCV%202019-green.svg)](https://mmact19.github.io/challenge/)

A software-only, reproducible multimodal Human Activity Recognition (HAR) framework pairing video clips with synchronized wearable inertial sensor time series from the [MMAct dataset](https://mmact19.github.io/challenge/).

---

## 1. Project Overview & Key Results

HAR-MMAct is a software-only multimodal Human Activity Recognition framework combining video features and wearable inertial sensor time series from the genuine MMAct dataset. It classifies 10 verified activity classes.

### Validation Benchmark
- **Selected validation model:** `exp1_multimodal` — MobileNetV2 + InceptionTime late fusion.
- **Validation accuracy:** 81.15% on 780 samples.
- **Validation Macro F1:** 0.8058.
- Model selection was based on validation performance, not held-out test performance.

### Final Held-Out Test Comparison

All three modality configurations were evaluated on the same 852 subject-independent held-out test samples. Their prediction files were checked for identical MMAct `sample_id` values and ordering.

| Model | Test Accuracy | Macro F1 | Macro Precision | Macro Recall |
|---|---:|---:|---:|---:|
| Video-only MobileNetV2 | 42.49% | 0.4317 | 0.4665 | 0.4552 |
| Sensor-only InceptionTime | 69.95% | 0.6665 | 0.7102 | 0.6987 |
| **Multimodal MobileNetV2 + InceptionTime** | **76.76%** | **0.7545** | **0.7843** | **0.7652** |

The multimodal model also achieved **0.9661 ROC-AUC (OvR)** and **0.8590 macro average precision** on the held-out test set.

### Main Findings
- Sensor-only performance was substantially stronger than video-only performance in these experiments.
- Multimodal fusion improved test accuracy by 6.81 percentage points over the sensor-only model.
- Multimodal fusion improved test accuracy by 34.27 percentage points over the video-only model.
- Performance remains class-dependent. Jumping is a difficult class for all three models.

Detailed metrics, per-class comparisons, prediction files, and plots are listed in the artifact index below.

---

## 2. Dataset Downloads

The project processes trimmed video clips and multi-modal wearable sensor streams. Because raw video and sensor archives exceed standard Git file limits, dataset files are hosted separately on Kaggle and are **not** committed to the GitHub repository.

### Kaggle Datasets:

1. **[Trimmed Scene Video Dataset](https://www.kaggle.com/datasets/vikramche/trimmed-scene-video)**
   *Description:* Kaggle-hosted video dataset prepared for the project's video modality. Contains trimmed MP4 video clips organized by subject, camera view, scene, and session.
2. **[MMAct Trimmed Dataset](https://www.kaggle.com/datasets/vikramche/mmact-trimmed)**
   *Description:* Kaggle-hosted trimmed MMAct dataset used as the project's dataset source. Contains 4 wearable inertial sensor streams (`acc_phone_clip`, `acc_watch_clip`, `gyro_clip`, `orientation_clip`), official split configuration files, and baseline trimmed clips.

> [!IMPORTANT]
> Users must download both datasets from Kaggle and place or extract them into the directory structure expected by the data-loading configuration described below.

---

## 3. Dataset Setup & Placement Instructions

### How to Download from Kaggle:
1. Log in to your [Kaggle](https://www.kaggle.com/) account.
2. Visit the dataset pages linked above and click **Download** (or use the Kaggle CLI: `kaggle datasets download -d vikramche/trimmed-scene-video` and `kaggle datasets download -d vikramche/mmact-trimmed`).
3. Extract the downloaded archives into the project's canonical dataset directory: `data/genuine_mmact/`.

### Expected Local Directory Layout:
The data loader ([`src/data/indexer.py`](src/data/indexer.py)) and stager ([`src/data/stager.py`](src/data/stager.py)) expect files placed under `data/genuine_mmact/`:

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

### Verification:
Verify your local dataset setup using the project audit tool:
```powershell
python scripts/audit_dataset.py
```
If your dataset is located in a different path, supply `--data_root <path>` to any CLI script or update `raw_data_root` in [`src/config.py`](src/config.py).

---

## 4. 10 Verified Target Activity Classes

Cross-referenced and verified against official challenge validation ground truth files (`trimmed_val_scene_gt.json`, `trimmed_val_view_gt.json`) and the ICCV 2019 publication (see [`docs/LABEL_MAPPING.md`](docs/LABEL_MAPPING.md)):

| Class Index | Target Class | Physical Description | Primary Sensor Modality |
|:---:|---|---|---|
| `0` | `crouching` | Elevation drop, postural change | Thigh vertical acceleration |
| `1` | `jumping` | Rapid upward thrust & landing shock | High-amplitude acceleration peaks |
| `2` | `kicking` | Rapid unilateral leg extension | Thigh/pocket accelerometer jerk |
| `3` | `pointing` | Sustained directional arm extension | Wrist smartwatch orientation |
| `4` | `running` | High-cadence periodic locomotion | High-frequency oscillatory gait |
| `5` | `sitting` | Static seated posture | Stable horizontal gravity component |
| `6` | `standing` | Static upright posture | Stable vertical gravity vector ($\approx 9.8\text{ m/s}^2$) |
| `7` | `using_phone` | Hand manipulation near torso | Fine rotational wrist gyro dynamics |
| `8` | `walking` | Regular gait locomotion | Standard 1–2 Hz gait cycles |
| `9` | `waving_hand` | Cyclic wrist oscillation | High-frequency periodic angular velocity |

---

## 5. Model Architectures

```
 [Video: (B, 3, 16, 112, 112)]        [Sensors: (B, 9, 100)]
               │                                │
     MobileNetV2 (Pretrained)         InceptionTime 1D CNN
               │                                │
      Dual Temporal Pooling             Global Average Pool
               │                                │
      Video Vector (256-d)             Sensor Vector (256-d)
               └───────────────┬────────────────┘
                               │
                       Late Concatenation
                            (512-d)
                               │
                        MLP Classifier
                  (Linear 128 -> BN -> ReLU)
                               │
                       Logits (10 classes)
```

1. **Video Branch (`MobileNetV2VideoBranch`):**
   - ImageNet-pretrained inverted residual blocks extracting 1280-dim per-frame features.
   - Dual temporal pooling (concatenated temporal Mean + Max pooling: 2560-dim $\rightarrow$ 256-dim).
   - Uniform sampling of $T=16$ frames per clip, normalized with ImageNet mean and std.
2. **Sensor Branch (`InceptionTimeSensorBranch`):**
   - 1D InceptionTime architecture matching official MMAct challenge conventions (Fawaz et al., 2020).
   - Multi-scale 1D convolutions (kernel sizes 9, 19, 39), bottleneck projection (32 filters), and residual connections across 9 inertial channels (`acc_phone`, `acc_watch`, `gyro`).
   - Aligned onto 100 Hz timeline via official backward-fill resampling (`bfill().dropna()`) into centered 100-sample (1.0 s) windows.
3. **Multimodal Late Fusion (`MultimodalHARModel`):**
   - Concatenates 256-dim video and 256-dim sensor embeddings into a 512-dim joint representation.
   - Classification head: Linear(128) $\rightarrow$ BatchNorm1d $\rightarrow$ ReLU $\rightarrow$ Dropout(0.35) $\rightarrow$ Linear(10).
   - Auxiliary classification heads allow isolated **video-only** and **sensor-only** inference from the same model architecture.

---

## 6. Validation Benchmark Comparison (All 9 Evaluated Models)

All 9 model configurations were evaluated on the identical validation partition ($N=780$ samples, 124 recordings) over 50 epochs on a Tesla T4 GPU with seed 42. Rankings are ordered by Validation Macro-F1 in descending order. Metrics match [`outputs/benchmark_validation/`](outputs/benchmark_validation/) and [`outputs/improved_cnn_v2/`](outputs/improved_cnn_v2/) exactly:

| Rank | Model Experiment | Modality | Video Backbone | Sensor Backbone | Val Acc (%) | Macro F1 | Precision | Recall | Selection Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **`exp1_multimodal`** | **Multimodal** | **MobileNetV2** | **InceptionTime** | **81.15%** | **0.8058** | **0.8657** | **0.8066** | **Top Model (Selected)** |
| 2 | `multimodal_fusion` | Multimodal | MobileNetV2 | InceptionTime | 80.38% | 0.8005 | 0.8509 | 0.7987 | Close Alternative |
| 3 | `tcn_gated_fusion` | Multimodal | MobileNetV2 | TCN 1D | 77.69% | 0.7643 | 0.8261 | 0.7711 | Gated Baseline |
| 4 | `tcn_sensor_only` | Sensor-Only | — | TCN 1D | 77.44% | 0.7538 | 0.7916 | 0.7684 | Top Sensor Baseline |
| 5 | `exp2_sensor_only` | Sensor-Only | — | InceptionTime | 77.44% | 0.7489 | 0.8219 | 0.7684 | Strong InceptionTime Baseline |
| 6 | `mobilenet_v2_staged_v2` | Video-Only | MobileNetV2 | — | 48.59% | 0.5026 | 0.5519 | 0.5106 | Staged Fine-Tuned |
| 7 | `mobilenet_v2` | Video-Only | MobileNetV2 | — | 35.26% | 0.3492 | 0.4340 | 0.3900 | Backbone Under-adapted |
| 8 | `improved_cnn_v2` | Video-Only | Custom CNN | — | 16.03% | 0.1171 | 0.0951 | 0.2013 | Scratch Retrained |
| 9 | `improved_cnn` | Video-Only | Custom CNN | — | 14.74% | 0.0981 | 0.1251 | 0.1733 | Scratch Optimization Stall |

---

## 7. Final Held-Out Test Evaluation Results

Three trained model configurations were evaluated using the same genuine MMAct subject-independent held-out test partition.

### 7.1 Test Protocol

- Training samples: 3,795
- Validation samples: 780
- Held-out test samples: 852
- Test recordings: 132
- Split strategy: subject-independent
- All three prediction files contain the same 852 `sample_id` values in identical order.
- The held-out test partition was not used for model training, hyperparameter tuning, or model selection.

### 7.2 Overall Test Metrics

| Metric | Video-only MobileNetV2 | Sensor-only InceptionTime | Multimodal Fusion |
|---|---:|---:|---:|
| Accuracy | 42.49% | 69.95% | **76.76%** |
| Macro F1 | 0.4317 | 0.6665 | **0.7545** |
| Macro Precision | 0.4665 | 0.7102 | **0.7843** |
| Macro Recall | 0.4552 | 0.6987 | **0.7652** |
| ROC-AUC (OvR) | 0.8443 | 0.9373 | **0.9661** |
| Macro Average Precision | 0.4732 | 0.7959 | **0.8590** |

### 7.3 Multimodal Per-Class Test Performance

| Activity | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
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

The weakest multimodal test class by F1 is `jumping` (0.2993). `running` also has low recall (0.4524), despite perfect precision. These weaknesses should be considered when interpreting the aggregate score.

### 7.4 Comparison Plots and Detailed Outputs

- [Test accuracy comparison](outputs/model_comparison/test_accuracy_comparison.png)
- [Test Macro F1 comparison](outputs/model_comparison/test_macro_f1_comparison.png)
- [Per-class F1 comparison](outputs/model_comparison/per_class_test_f1_comparison.png)
- [Complete model comparison CSV](outputs/model_comparison/test_metrics_comparison.csv)
- [Per-class metrics for all three models](outputs/model_comparison/per_class_test_metrics.csv)

The complete per-model evaluation JSON files, test predictions, confusion matrices, ROC curves, and precision-recall plots are preserved in their respective experiment directories.

---

## 8. Environment Setup & Reproducibility

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

## 9. Inference & Demo Execution

The standalone inference CLI ([`scripts/infer.py`](scripts/infer.py)) supports three verified prediction modes based on input arguments:
1. **Full Multimodal Inference (Video + Wearable Sensors):** Fuses video frames and 9-channel inertial sensor series via the late fusion layer.
2. **Video Stream Alone:** Uses the video branch (MobileNetV2 feature extractor) with the auxiliary video classifier head of the multimodal architecture.
3. **Sensor Stream Alone:** Uses the sensor branch (InceptionTime 1D CNN) with the auxiliary sensor classifier head of the multimodal architecture.

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
> **Inference Demo Terminology & Verification Scope:** The test recorded above ran `scripts/infer.py` on `challenge-main/challenge-main/utils/sample_clip/carrying.mp4` using the selected multimodal checkpoint (`outputs/exp1_multimodal/checkpoints/best_model.pth`). Because no sensor files were provided, the CLI operated strictly in **video-stream-only mode** via the auxiliary video head of the multimodal architecture. This verifies standalone video-branch inference, but does not represent end-to-end verification of paired multimodal (video + sensor) input data.

---

## 10. Artifact Index

### Consolidated Comparison

| Artifact | Path |
|---|---|
| Test metric comparison | `outputs/model_comparison/test_metrics_comparison.csv` |
| Machine-readable comparison | `outputs/model_comparison/test_metrics_comparison.json` |
| Per-class metrics | `outputs/model_comparison/per_class_test_metrics.csv` |
| Accuracy comparison graph | `outputs/model_comparison/test_accuracy_comparison.png` |
| Macro F1 comparison graph | `outputs/model_comparison/test_macro_f1_comparison.png` |
| Per-class F1 graph | `outputs/model_comparison/per_class_test_f1_comparison.png` |

### Video-only MobileNetV2

- Evaluation: `outputs/mobilenet_v2_staged_v2_retrained/test_evaluation_v2.json`
- Predictions with sample IDs: `outputs/mobilenet_v2_staged_v2_retrained/test_predictions.csv`
- Plots: `outputs/mobilenet_v2_staged_v2_retrained/test_plots_v2/`

### Sensor-only InceptionTime

- Evaluation: `outputs/sensor_only_inception_time_v2/test_evaluation.json`
- Predictions with sample IDs: `outputs/sensor_only_inception_time_v2/test_predictions.csv`
- Plots: `outputs/sensor_only_inception_time_v2/test_plots/`

### Multimodal MobileNetV2 + InceptionTime

- Evaluation: `outputs/exp1_multimodal/test_evaluation_v2.json`
- Predictions with sample IDs: `outputs/exp1_multimodal/test_predictions.csv`
- Plots: `outputs/exp1_multimodal/test_plots_v2/`

### Training and Reproducibility

- Validation benchmark outputs: `outputs/benchmark_validation/`
- Multimodal training history: `outputs/exp1_multimodal/history.json`
- Sensor-only training summary: `outputs/sensor_only_inception_time_v2/training_summary.json`
- Sensor-only runtime report: `outputs/sensor_only_inception_time_v2/runtime_report.json`
- Trained Kaggle notebook: `notebooks/ML_Trained_book/mmact-har-ml-model_Trained.ipynb`

> [!NOTE]
> Cache directories and temporary preprocessing files are not part of the result handover. Model checkpoint binaries are subject to the repository's Git ignore rules; check the repository's checkpoint availability notes before attempting inference from a fresh clone.

---

## 11. Kaggle Cloud GPU Execution

For execution on Kaggle GPU:
1. Open [`notebooks/ML_Trained_book/mmact-har-ml-model_Trained.ipynb`](notebooks/ML_Trained_book/mmact-har-ml-model_Trained.ipynb) or generate a fresh notebook with `python scripts/create_kaggle_notebook.py`.
2. Attach both Kaggle datasets:
   - `vikramche/trimmed-scene-video`
   - `vikramche/mmact-trimmed`
3. Enable GPU accelerator (`Tesla T4 x2` or `P100`) and set Internet to `On`.
4. Run all cells top-to-bottom. The automated staging discovers attached inputs, verifies integrity, caches preprocessed tensors, trains all benchmark models, and exports `har_mmact_results.zip`.

### Individual Training vs Held-Out Test Accuracy

These checkpoint-level plots compare the training accuracy recorded
during each selected checkpoint epoch with the corresponding held-out
test accuracy. Test data is evaluated after model selection, not once
per epoch.

- Video-only: `outputs/mobilenet_v2_staged_v2_retrained/plots/train_vs_test_accuracy_checkpoint.png`
- Sensor-only: `outputs/sensor_only_inception_time_v2/plots/train_vs_test_accuracy_checkpoint.png`
- Multimodal: `outputs/exp1_multimodal/plots/train_vs_test_accuracy_checkpoint.png`
- Combined comparison: `outputs/model_comparison/train_vs_test_accuracy_final_models.png`
- Source script: `scripts/plot_train_vs_test_accuracy.py`
- Data table: `outputs/model_comparison/train_vs_test_accuracy_checkpoint.csv`
