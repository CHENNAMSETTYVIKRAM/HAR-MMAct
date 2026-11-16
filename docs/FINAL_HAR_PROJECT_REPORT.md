# Human Activity Recognition Using Multimodal MMAct Data

## 1. Executive summary

This project implements a software-only multimodal Human Activity
Recognition system using video and wearable-sensor data from MMAct.

The system evaluates ten target activities. It includes video-only,
sensor-only and multimodal deep-learning architectures.

The best evaluated model is **Primary Multimodal MobileNetV2 +
InceptionTime**, which achieved 76.76% test accuracy and 0.7545
test Macro F1 on the project's held-out subject-independent test set.

## 2. Dataset and target classes

The staged dataset contained:

- 17,820 video files.
- 35,685 sensor CSV files.
- 20 subjects.
- Four discovered sensor modalities:
  `acc_phone_clip`, `acc_watch_clip`, `gyro_clip`,
  and `orientation_clip`.
- 5,427 valid paired video-sensor samples.

The model's sensor branch uses nine channels:
three phone accelerometer channels, three watch accelerometer
channels and three gyroscope channels. Orientation files were
discovered but were not used as model input.

The ten target activities are:

1. Crouching
2. Jumping
3. Kicking
4. Pointing
5. Running
6. Sitting
7. Standing
8. Using phone
9. Walking
10. Waving hand

## 3. Preprocessing and split

Video clips were represented using 16 sampled frames per sample.
The video branch used 112 × 112 frame inputs.

The custom subject-independent partition was:

| Split | Subjects | Samples |
|---|---|---:|
| Train | Subjects 1–14 | 3,795 |
| Validation | Subjects 15–17 | 780 |
| Test | Subjects 18–20 | 852 |
| Total | 20 subjects | 5,427 |

Video, recording, session and subject isolation checks passed.

**Protocol limitation:** This is the project's custom subject-independent
split. It is not the official MMAct cross-scene/cross-view challenge
protocol. Results should not be presented as official challenge results.

## 4. Model benchmark

All ten models were compared using the held-out test results.

|   test_rank_by_macro_f1 | model                                          | modality            | accuracy   |   precision |   recall |   macro_f1 |   roc_auc_ovr |   macro_ap |
|------------------------:|:-----------------------------------------------|:--------------------|:-----------|------------:|---------:|-----------:|--------------:|-----------:|
|                       1 | Primary Multimodal MobileNetV2 + InceptionTime | Multimodal          | 76.76%     |      0.7843 |   0.7652 |     0.7545 |        0.9661 |     0.859  |
|                       2 | Additional Multimodal Fusion                   | Multimodal          | 75.47%     |      0.777  |   0.7538 |     0.7464 |        0.9662 |     0.85   |
|                       3 | Gated Multimodal TCN                           | Multimodal          | 73.94%     |      0.7699 |   0.7392 |     0.7241 |        0.9582 |     0.8268 |
|                       4 | Sensor-only TCN                                | Sensor-only         | 74.65%     |      0.8133 |   0.7429 |     0.719  |        0.9524 |     0.8094 |
|                       5 | Sensor-only InceptionTime                      | Sensor-only         | 68.54%     |      0.707  |   0.681  |     0.6563 |        0.9311 |     0.7742 |
|                       6 | Retrained MobileNetV2                          | Video-only          | 42.49%     |      0.4665 |   0.4552 |     0.4317 |        0.8443 |     0.4732 |
|                       7 | Original MobileNetV2                           | Video-only          | 42.14%     |      0.4473 |   0.459  |     0.4153 |        0.8609 |     0.4982 |
|                       8 | MobileNetV2 + TCN                              | Video-only temporal | 33.57%     |      0.3277 |   0.3716 |     0.3277 |        0.8053 |     0.3699 |
|                       9 | Original Custom CNN                            | Video-only          | 15.38%     |      0.1541 |   0.1727 |     0.1139 |        0.616  |     0.1791 |
|                      10 | Retrained Custom CNN V2                        | Video-only          | 13.26%     |      0.1307 |   0.1449 |     0.094  |        0.6091 |     0.1786 |

## 5. Best model

The primary multimodal MobileNetV2 + InceptionTime model achieved:

| Metric | Test result |
|---|---:|
| Accuracy | 76.76% |
| Macro precision | 0.7843 |
| Macro recall | 0.7652 |
| Macro F1 | 0.7545 |
| ROC-AUC (OvR macro) | 0.9661 |
| Macro average precision | 0.8590 |

This is the strongest evaluated model in the ten-model benchmark.

## 6. Additional video experiments

Additional experiments evaluated:

- MobileNetV2 Fine-tuning V2.
- MobileNetV2 + BiLSTM.
- MobileNetV2 + TCN.

The TCN temporal head was the strongest newly trained temporal-head
candidate by validation Macro F1 (0.3432). Its held-out test
accuracy was 33.57%, with Macro F1 of 0.3277.

It did not improve the original MobileNetV2 video-only baseline
or the primary multimodal model.

Detailed results are in `ADDITIONAL_VIDEO_EXPERIMENTS.md`.

## 7. Evaluation methodology

The project reports accuracy, macro precision, macro recall,
Macro F1, multiclass ROC-AUC using one-vs-rest probabilities,
and macro average precision.

Model selection used validation results. The held-out test partition
was kept separate from training and validation-based model selection.

The TCN test results were computed from the selected TCN checkpoint
and the corresponding fine-tuned MobileNetV2 feature extractor.

## 8. Limitations

- The dataset contains a limited number of subjects and paired samples.
- Some activities remain harder to classify than others.
- Video-only models performed below the best sensor-only and
  multimodal models.
- The three-trial hyperparameter search was limited, not exhaustive.
- The new TCN and BiLSTM experiments trained temporal heads using
  frozen MobileNetV2 features.
- The reported split is custom subject-independent rather than
  the official MMAct challenge protocol.
- Do not claim accuracy or runtime beyond the measured experiments.
- The available inference demo was previously verified in video-only
  mode; do not claim full multimodal deployment without testing it.

## 9. Reproducibility and artifacts

The experiment archive includes model checkpoints, evaluation
metrics, predictions, plots, hyperparameter trial outputs and
comparison results where available.

The archive manifest and SHA-256 checksum should be retained with
the downloaded experiment package.

## 10. Conclusion

The experiments show that combining video and wearable-sensor
features provides the strongest result among the evaluated
architectures in this project.

The primary multimodal MobileNetV2 + InceptionTime model remains
the selected project model. The additional video temporal models
are retained as comparative experiments, not claimed improvements.
