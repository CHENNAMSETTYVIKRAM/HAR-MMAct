# HAR-MMAct — Final Experimental Report

## Dataset and evaluation protocol

- Dataset: Genuine MMAct paired video and wearable sensor data.
- Target activities: 10 project activity classes.
- Split: Subject-independent train/validation/test.
- Training: 50 epochs per model.
- Validation samples: 780.
- Held-out test samples: 852.
- Test data was kept separate from model selection.

## Nine-model results

| Model                  |   Best Validation Accuracy (%) |   Test Accuracy (%) |   Test Macro F1 |
|:-----------------------|-------------------------------:|--------------------:|----------------:|
| exp1_multimodal        |                        81.1538 |             77.23   |        0.760222 |
| multimodal_fusion      |                        80.3846 |             75.59   |        0.7482   |
| tcn_gated_fusion       |                        77.6923 |             73.9437 |        0.72414  |
| tcn_sensor_only        |                        77.4359 |             74.6479 |        0.719047 |
| exp2_sensor_only       |                        77.9487 |             68.54   |        0.6578   |
| mobilenet_v2_staged_v2 |                        49.359  |            nan      |      nan        |
| mobilenet_v2           |                        35.2564 |             42.14   |        0.4153   |
| improved_cnn_v2        |                        16.7949 |            nan      |      nan        |
| improved_cnn           |                        14.8718 |             15.38   |        0.1139   |

## Hyperparameter tuning

Four configurations were evaluated using train/validation data only.
Trial 1 achieved the highest validation accuracy (15.90%).
Trial 2 achieved the highest validation Macro F1 (0.0815).
Trial 2 was selected for the tuned CNN training using Macro F1.
The tuned CNN training completed 50 epochs and achieved best validation Macro F1 of 0.1118.

## Transfer learning

MobileNetV2 staged fine-tuning was implemented.
The staged experiment used a frozen-backbone phase followed by fine-tuning.
Its best validation accuracy was 49.36%.
Its checkpoint is not currently available for held-out test evaluation.

## Best measured test model

Exp1 multimodal achieved 77.23% held-out test accuracy,
0.7602 macro F1 and 0.9673 multiclass ROC-AUC (one-vs-rest).

## Limitations

- V2 model checkpoints are currently missing from the active Kaggle runtime.
- Their validation metrics are available, but held-out test metrics are not.
- Detailed test ROC/PR metrics are available only for models with saved detailed test evaluation JSON.
- Video-only custom CNN models performed poorly and require architectural/data-pipeline investigation.
- The modality comparison represents independently trained models, not a controlled ablation.

## Reproducibility

Training histories, validation predictions, experiment manifests,
evaluation JSON files and plots should be retained with the project.
Do not claim unmeasured test results.