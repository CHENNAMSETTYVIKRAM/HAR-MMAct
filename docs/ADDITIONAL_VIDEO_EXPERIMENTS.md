# Additional Video-Only Experiments

**Experiment date:** 16 November 2026

## Objective

Evaluate whether staged MobileNetV2 fine-tuning and temporal
classification heads improve the existing video-only branch.

## Validation results

| Model | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| Original MobileNetV2 | 35.26% | 0.3492 |
| Previous Staged MobileNetV2 | 36.54% | Not available |
| MobileNetV2 Fine-tuning V2 | 30.13% | 0.2932 |
| MobileNetV2 + BiLSTM | 31.41% | 0.3216 |
| MobileNetV2 + TCN | 34.74% | 0.3432 |

The original MobileNetV2 had the highest available validation Macro F1.
Among the newly developed temporal-head experiments, TCN performed best.

## New TCN experiment

The temporal TCN head received 16 ordered frame features,
each with 1,280 dimensions, extracted from the fine-tuned
MobileNetV2 backbone.

The MobileNetV2 feature extractor was frozen during temporal-head
training. This was a feature-extraction experiment, not end-to-end
joint training of the complete video model.

- Best checkpoint epoch: 9
- Validation Macro F1: 0.3432
- Test samples: 852

### Held-out test metrics

| Metric | Result |
|---|---:|
| Accuracy | 33.57% |
| Macro precision | 0.3277 |
| Macro recall | 0.3716 |
| Macro F1 | 0.3277 |
| ROC-AUC (OvR macro) | 0.8053 |
| Macro average precision | 0.3699 |
| Test loss | 2.1416 |

### Interpretation

The TCN was the strongest of the two new temporal-head experiments,
but it did not improve the original video-only benchmark.

It should be reported as an additional experiment rather than
as a performance improvement.

The original MobileNetV2 test benchmark recorded 42.14% accuracy
and 0.4153 Macro F1. The new TCN recorded 33.57% accuracy
and 0.3277 Macro F1.

## Limitations

- Hyperparameter screening used only three configurations.
- The temporal heads used frozen extracted MobileNetV2 features.
- Video-only performance remains below the multimodal and
  sensor-only benchmark models.
- Results use the project's custom subject-independent split,
  not the official MMAct cross-scene/cross-view challenge protocol.
