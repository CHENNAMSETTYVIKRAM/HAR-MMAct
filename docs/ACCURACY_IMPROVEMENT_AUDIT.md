# HAR-MMAct Video Model Accuracy Improvement Audit

## Scope

This audit is restricted to the four existing benchmark entries:

- Original MobileNetV2: 42.1362% held-out test accuracy
- Retrained MobileNetV2: 42.4883% held-out test accuracy
- Original Custom CNN: 15.3756% held-out test accuracy
- Retrained Custom CNN V2: 13.2629% held-out test accuracy

No source files or experiment artifacts were modified during this audit. No training was started. No held-out test artifact was used to recommend or tune a configuration; held-out test data is used here only to report the existing benchmark outcomes and their existing per-class evidence.

## Executive conclusions

### Confirmed repository facts

1. All four benchmarks use the same 10-class label mapping and the same subject-independent split membership checks.
2. All four are video-only runs of `MultimodalHARModel` with `override_mode="video_only"`.
3. The video preprocessing is deterministic uniform 16-frame sampling followed by 112×112 resize and ImageNet normalization.
4. There is no repository-defined video augmentation in `VideoLoader` or the dataset path.
5. The class-weighted `CrossEntropyLoss` is enabled by default and was used unless a run explicitly disabled it.
6. Checkpoints are selected by best validation Macro F1, not validation accuracy.
7. Split isolation is reported as passing for the subject split in every inspected manifest.
8. The Custom CNN histories show little separation from chance and strong class collapse; the MobileNet histories show substantial train/validation disagreement and unstable validation loss.
9. The retrained MobileNetV2 improved validation accuracy from 35.26% to 36.54%, a 1.28 percentage-point increase, while the test value changed only from 42.1362% to 42.4883%.
10. The retrained Custom CNN V2 validation accuracy decreased from 14.74% to 13.46% in the run manifest; it also failed to improve over the original Custom CNN benchmark.

### Not confirmed as repository bugs

- No label-to-index mismatch is indicated by the repository: the indexer maps all target names through `TARGET_CLASS_TO_IDX`, and manifests list 10 contiguous target classes.
- No video/sample shape bug is visible in the forward-path code or cache validator.
- No split leakage was reported for subject isolation.
- Pretrained MobileNetV2 loading is supported correctly at the code level, but the artifact manifests do not preserve enough runtime output to prove or disprove weight-loading behavior for every run.
- There is no evidence of a data-loader cache-corruption bug affecting these runs. The cache validator checks 4D video shape and sensor shape and rejects NaN/Inf tensors.

## Shared dataset and preprocessing evidence

### Classes

The target mapping is declared in [src/config.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/config.py#L19-L38):

| Index | Class |
|---:|---|
| 0 | crouching |
| 1 | jumping |
| 2 | kicking |
| 3 | pointing |
| 4 | running |
| 5 | sitting |
| 6 | standing |
| 7 | using_phone |
| 8 | walking |
| 9 | waving_hand |

`MMActIndexer` filters to these classes and derives `label_idx` directly from `TARGET_CLASS_TO_IDX` in [src/data/indexer.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/data/indexer.py#L246-L410).

### Split

All four audited runs use `split_strategy="subject"`:

- Original manifests: 3795 train, 780 validation, 852 test.
- Retrained manifests: 3799 train, 780 validation, 852 test.
- Subject partitions are train subjects 1–14, validation subjects 15–17, and test subjects 18–20 in [src/config.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/config.py#L103-L122).

The later retrained runs contain four additional training samples. Their manifests report recording-disjoint, session-disjoint, subject-disjoint, and video-disjoint train/validation/test checks as passed. These results are consistent with the split implementation in [scripts/train.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/scripts/train.py#L288-L317), but the training artifacts do not expose the underlying record lists for independent text audit.

### Class distribution

The original run manifests report:

- crouching 375
- jumping 375
- kicking 375
- pointing 375
- running 375
- sitting 220
- standing 355
- using_phone 595
- walking 375
- waving_hand 375

The retrained runs report 359 standing instead of 355 and 3799 rather than 3795 train samples; validation and test distributions remain 780 and 852 respectively. The imbalance is moderate and concentrated in `using_phone` and, to a lesser extent, `sitting`.

Sources: [Original MobileNet manifest](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/mobilenet_v2/experiment_manifest.json) and [retrained MobileNet manifest](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/mobilenet_v2_staged_v2_retrained/experiment_manifest.json).

### Frame sampling

[VideoLoader](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/data/video_loader.py#L21-L108):

- Reads 16 uniformly spaced indices across `CAP_PROP_FRAME_COUNT`.
- Repeats the final frame if a short/failed clip does not provide 16 frames.
- Falls back to sequential decode and uniform sampling if frame count metadata is unreliable.
- Resizes every sampled frame to 112×112 using bilinear interpolation.
- Produces `(C, T, H, W)` tensors.

The default configuration is in [src/config.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/config.py#L69-L83): 16 frames, 112×112, uniform strategy.

There is no dataset-side frame augmentation or temporal crop variation.

### Normalization

Frames are scaled to `[0,1]`, then normalized with mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)` in [src/data/video_loader.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/data/video_loader.py#L110-L124). This matches torchvision ImageNet normalization.

No random resize crop, color jitter, horizontal flip, MixUp, CutMix, or other augmentation is implemented in `VideoLoader` or `MMActMultimodalDataset`.

### Loss and checkpoint selection

Training uses AdamW and inverse-frequency class-weighted `CrossEntropyLoss` by default in [scripts/train.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/scripts/train.py#L399-L425).

The best checkpoint is saved only when validation Macro F1 increases in [scripts/train.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/scripts/train.py#L582-L593). This means the best checkpoint is not necessarily the epoch with the highest validation accuracy.

## 1. Original MobileNetV2

### Benchmark

- Test accuracy: 42.1362%
- Test Macro F1: 0.4153
- Validation accuracy at best checkpoint: 35.2564%
- Validation Macro F1: 0.3492174

### Architecture and source

`MobileNetV2VideoBranch` is implemented in [src/models/video_branch.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/models/video_branch.py#L254-L334).

It:

1. Uses torchvision MobileNetV2 features when `pretrained=True` and no external pretrained path is supplied.
2. Falls back to the repository's pure-Python MobileNetV2 implementation if torchvision initialization fails.
3. Flattens frame dimensions to `(B*T, C, H, W)`.
4. Applies 1280-dimensional per-frame average pooling.
5. Computes mean and max temporal pooling.
6. Projects 2560 temporal features to a 128-dimensional embedding using Dropout, Linear, BatchNorm1d, and ReLU.

The video-only classifier in [src/models/fusion_model.py](https://github.com/CHENNAMSETTYVIKMOVIKRAM/HAR-MMAct/blob/master/src/models/fusion_model.py#L118-L126) is:

```text
Linear(128, 128)
BatchNorm1d(128)
ReLU
Dropout(0.35)
Linear(128, 10)
```

### Training configuration

From [outputs/mobilenet_v2/experiment_manifest.json](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/mobilenet_v2/experiment_manifest.json):

- Batch size: 8
- Learning rate: 1e-4
- Weight decay: 5e-4
- Dropout: 0.35
- 50 epochs
- Subject-independent split
- AMP enabled
- AdamW
- CosineAnnealingLR
- CrossEntropyLoss with inverse-frequency class weights

### Frozen/trainable status

The training source creates the full fusion model with `pretrained_video=True` and `freeze_video_backbone=args.freeze_video or args.staged_finetune` in [scripts/train.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/scripts/train.py#L374-L392).

The original manifest does not list the freeze/staged flags. The manifest's history is not sufficient to prove the exact freeze transition; for example, peak GPU memory rises at epoch 15, but this could be consistent with staged fine-tuning or backend allocation behavior.

The source supports:

- Full parameter training when neither `--freeze_video` nor `--staged_finetune` is used.
- Backbone freezing when `--freeze_video` is used.
- Two-stage staged fine-tuning, where the backbone is frozen until `--unfreeze_epoch` and then unfrozen with a separate lower learning rate in [scripts/train.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/scripts/train.py#L498-L512).

The exact original run command and complete parameter count are missing from the persisted artifacts.

### History

The 50-epoch history is in [outputs/mobilenet_v2/history_mobilenet_v2.csv](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/mobilenet_v2/history_mobilenet_v2.csv).

- Training accuracy rises from 13.70% to 51.54%.
- Validation accuracy peaks at 35.26% at epoch 41.
- Validation loss is more variable than training loss and does not converge monotonically.
- Final validation accuracy is 29.62%, substantially below the selected epoch 41 value.

This is a confirmed generalization gap: the late run continues to improve training accuracy while validation accuracy remains unstable and far lower.

### Per-class validation evidence

The existing best validation summary shows [sitting](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/mobilenet_v2/training_summary_mobilenet_v2.json#L11-L67):

- sitting: F1 0.7500
- crouching: F1 0.5385
- running: F1 0.5167
- walking: F1 0.4815
- jumping: F1 0.1111
- kicking: F1 0.1682
- waving_hand: F1 0.0875

The confusion matrix shows broad confusion for jumping, kicking, pointing, standing, using_phone, and waving_hand.

### Checkpoint and loading

The artifact is at [outputs/mobilenet_v2/checkpoints/best_model_mobilenet_v2.pth](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/mobilenet_v2/checkpoints/best_model_mobilenet_v2.pth). Loading is implemented with `model.load_state_dict(checkpoint["model_state_dict"])` in evaluation code; the exact evaluation script should be checked against the training model constructor for architecture-argument consistency.

### Assessment

Likely causes supported by evidence:

1. High 16-frame mean/max temporal pooling may lose local motion ordering and can favor static cues; this is a limitation of the representation, not a confirmed bug.
2. The run ended with significant training/validation disagreement.
3. Uniform sampling may not cover all meaningful transient motion when clips have unequal durations.
4. No augmentation is present.
5. The exact ImageNet pretrained-weight initialization and freeze status need runtime-level verification.

No source evidence proves an incorrect label mapping or incorrect split.

## 2. Retrained MobileNetV2

### Benchmark

- Test accuracy: 42.4883%
- Test Macro F1: 0.4316926
- Validation accuracy at best selected checkpoint: 36.5385%
- Validation Macro F1: 0.3669264

### Architecture

The architecture is still `MobileNetV2VideoBranch`, with the same temporal mean+max pooling and 128-dimensional video embedding. It uses the video-only head in `MultimodalHARModel`; source: [src/models/video_branch.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/models/video_branch.py#L254-L334) and [src/models/fusion_model.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/models/fusion_model.py#L118-L126).

### Training configuration

[Retrained manifest](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/mobilenet_v2_staged_v2_retrained/experiment_manifest.json):

- Batch size: 8
- Learning rate: 0.0002
- Weight decay: 0.0001
- Dropout: 0.20
- 50 requested epochs; 44 completed epochs
- Staged fine-tuning metadata is not stored in the manifest
- AMP enabled
- AdamW
- CosineAnnealingLR
- 3799 train samples, 780 validation samples, 852 test samples

### Stages and frozen parameters

The `lr` history in [history_mobilenet_v2.csv](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/mobilenet_v2_staged_v2_retrained/history_mobilenet_v2.csv) shows approximately 2e-4 for epochs 1–14, then approximately 1e-5 for epochs 15 onward. The peak GPU memory rises from 328.2 MB to 1322.2 MB at epoch 15. This pattern is consistent with staged backbone unfreezing at epoch 15, but the manifest itself does not record `staged_finetune`, `freeze_video`, or `unfreeze_epoch`.

After unfreezing, the source should create two optimizer groups: backbone at `unfreeze_lr=1e-5` and all remaining trainable parameters at `lr=2e-2`? The source sets head params at `args.lr` and backbone at `args.unfreeze_lr` in [scripts/train.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/scripts/train.py#L498-L512). The persisted history loses the exact per-parameter learning rates.

Before unfreezing, MobileNet backbone parameters are frozen and only the classifier/head should be trainable. After unfreezing, all backbone and head parameters are trainable.

### History

The history has 44 entries in [outputs/mobilenet_v2_staged_v2_retrained/history_mobilenet_v2.csv](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/mobilenet_v2_staged_v2_retrained/history_mobilenet_v2.csv):

- Training accuracy rises from 16.74% to 63.10% by epoch 44.
- Validation accuracy peaks at 36.54% at epoch 32.
- Validation loss reaches a minimum around 1.6717 at epoch 32, then rises sharply later.
- The training/validation gap continues to widen after epoch 32.

This run is evidence for overfitting or a poor fine-tuning schedule, not a confirmed bug.

### Per-class validation evidence

The best validation summary reports:

- sitting F1: 0.7603
- crouching F1: 0.5294
- running F1: 0.4956
- walking F1: 0.4460
- jumping F1: 0.2879
- kicking F1: 0.1284
- waving_hand F1: 0.0885

The confusion matrix remains weak for kicking, pointing, using_phone, and waving_hand. The full summary is [outputs/mobilenet_v2_staged_v2_retrained/training_summary_mobilenet_v2.json](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/mobilenet_v2_staged_v2_retrained/training_summary_mobilenet_v2.json#L11-L193).

### Checkpoint selection/loading

- Best checkpoint: [outputs/mobilenet_v2_staged_v2_retrained/checkpoints/best_model_mobilenet_v2.pth](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/mobilenet_v2_staged_v2_retrained/checkpoints/best_model_mobilenet_v2.pth)
- Runtime report: 44 completed epochs, 4294.44 seconds.
- Selection is by validation Macro F1, with peak F1 0.3669 in the runtime report.

### Assessment

Verified findings:

1. Pretrained weight loading is intended, but the artifact lacks a print/audit trail for successful torchvision weight load.
2. Lower dropout and staged fine-tuning did not materially increase validation accuracy relative to the original run.
3. Validation accuracy improved only 1.28 points while training accuracy increased dramatically.
4. There is a strong late generalization gap and validation loss regression after epoch 32.

Possible improvement directions:

- Verify normalization, pretrained weights, and backbone freeze/unfreeze state before tuning.
- Shorter fine-tuning horizon and frequent validation selection.
- Preserve more temporal information than mean+max pooling; this should be screened only if the frame-level diagnostics pass.
- Add only validated, non-destructive video augmentations after confirming the original data pipeline.

## 3. Original Custom CNN

### Benchmark

- Test accuracy: 15.3756%
- Test Macro F1: 0.1139180
- Validation accuracy at best checkpoint: 14.7435897%
- Validation Macro F1: 0.0980818

### Architecture and source

The original branch is `OriginalCNNVideoBranch`, alias `LightweightVideoBranch`, in [src/models/video_branch.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/models/video_branch.py#L337-L388).

Frame processing:

```text
Conv2d(3,32, stride=2) -> BatchNorm -> ReLU -> MaxPool
Conv2d(32,64, stride=2) -> BatchNorm -> ReLU -> MaxPool
Conv2d(64,128) -> ReLU -> AdaptiveAvgPool(1,1)
128-D frame features
Mean + max temporal pooling -> 256-D
Linear(256,128) -> BatchNorm1d -> ReLU
Dropout 0.35 in the video-only head
Linear(128,10)
```

The branch uses no spatial dropout and lower temporal dropout in the branch itself; the video-only head adds additional dropout.

### Training configuration

[Original Custom CNN manifest](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/improved_cnn/experiment_manifest.json):

- Batch size: 8
- Learning rate: 1e-4
- Weight decay: 0.001
- Dropout: 0.35
- 50 epochs
- 3795 train samples
- 780 validation samples
- 852 test samples
- AMP enabled
- AdamW
- CosineAnnealingLR
- Class weighting enabled by default

### Frozen/trainable status

The Custom CNN is randomly initialized and has no pretrained/frozen phase. All branch parameters and the `video_only_classifier` parameters are trainable. The full `MultimodalHARModel` also constructs and optimizes unused sensor/fusion parameters in video-only mode, because source training builds the complete model and AdamW includes every `requires_grad` parameter. This is confirmed from [scripts/train.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/scripts/train.py#L378-L397) and [scripts/train.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/scripts/train.py#L420-L425). The unused parameters do not receive gradients in `video_only` forward, but they increase model state and parameter count.

### History

The 50-epoch history is [outputs/improved_cnn/history_custom_cnn.csv](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/improved_cnn/history_custom_cnn.csv).

- Train accuracy stays near 15–17%.
- Validation accuracy oscillates roughly 9–15% and peaks near 14.87%.
- Training loss decreases slowly from 2.3909 to 2.0506.
- Validation loss remains around 2.15–2.40 and is volatile.

This is confirmed low learning, not merely a poor test split. The model has no evidence of learning beyond weak class biases.

### Per-class and confusion evidence

The held-out test result is concentrated in only a few classes:

- sitting F1: 0.5591
- jumping F1: 0.1988
- running F1: 0.1852
- using_phone F1: 0.1961
- crouching/kicking/pointing/standing/walking/waving_hand: F1 0.0000

The test confusion matrix shows most classes mapping into jumping or running, while sitting and using_phone have limited recognition. Source: [outputs/improved_cnn/final_test_evaluation.json](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/improved_cnn/final_test_evaluation.json#L25-L206).

### Checkpoint selection/loading

Checkpoint: [outputs/improved_cnn/checkpoints/best_model_cnn.pth](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/improved_cnn/checkpoints/best_model_cnn.pth).

The checkpoint is the model state with the best validation Macro F1. There is no artifact evidence of a shape mismatch or corrupted state dict.

### Assessment

Confirmed:

- The architecture is small, randomly initialized, and trained from scratch.
- The run does not learn a 10-class video representation effectively.
- Training is class-biased, and held-out class F1 confirms severe class collapse.

Likely causes supported by model and history evidence:

1. The 128×2 temporal representation may be too weak for the full 112×112 clip action-recognition task.
2. The 16 independent frame features and mean+max temporal pooling do not encode temporal order.
3. No video augmentation or pretrained visual initialization is used.
4. The very small 15–17% train accuracy indicates the training dynamics or representation have not found useful features.

A bug cannot be claimed without small-subset overfit and gradient evidence.

## 4. Retrained Custom CNN V2

### Benchmark

- Test accuracy: 13.2629%
- Test Macro F1: 0.0939734
- Validation accuracy at best checkpoint: 13.461538%
- Validation Macro F1: 0.0978963

### Architecture and source

This run uses `CustomCNNVideoBranch` from [src/models/video_branch.py](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/models/video_branch.py#L391-L447).

Compared with the original branch, it adds:

- Conv bias disabled.
- SpatialDropout2d 0.15 after the first stage.
- SpatialDropout2d 0.20 after the second stage.
- Same 128-dimensional per-frame features.
- Same mean+max temporal pooling.
- Same 128-dimensional temporal projection.
- Dropout in the projection and video-only head.

This is still a shallow randomly initialized 2D-CNN-per-frame representation; it is not an architecture change from the original CNN beyond stronger regularization.

### Training configuration

[Retrained Custom CNN manifest](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/improved_cnn_v2_retrained_v2/experiment_manifest.json):

- Batch size: 8
- Learning rate: 1e-4
- Weight decay: 0.0001
- Dropout: 0.35
- 50 requested epochs
- 30 completed epochs
- 3799 train samples
- 780 validation samples
- 852 test samples
- AMP enabled
- AdamW
- CosineAnnealingLR

The run stopped after 30 completed epochs. It did not complete all 50 requested epochs. The manifest's runtime report is 3837.98 seconds and reports 30 epochs completed.

### Frozen/trainable status

No pretrained/frozen stage applies. All Custom CNN and video-only classifier parameters are trainable. As in the original CNN run, the complete multimodal model is instantiated and therefore unused sensor/fusion parameters are included in the model and optimizer state, even though video-only forward does not use them.

### History

The history is [outputs/improved_cnn_v2_retrained_v2/history_custom_cnn.csv](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/improved_cnn_v2_retrained_v2/history_custom_cnn.csv):

- Training accuracy remains approximately 10–16%.
- Validation accuracy peaks around 14.36%.
- Validation loss is volatile and not steadily decreasing.
- The best checkpoint at epoch 17 has 13.46% validation accuracy and 0.0979 Macro F1.

The run has the clearest evidence of class collapse: in the validation summary, only sitting, using_phone, and waving_hand have non-zero F1, with crouching, kicking, pointing, running, and walking at 0.0. The full summary is [outputs/improved_cnn_v2_retrained_v2/training_summary_custom_cnn.json](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/improved_cnn_v2_retrained_v2/training_summary_custom_cnn.json).

### Per-class test evidence

The final test evaluation is [outputs/improved_cnn_v2_retrained_v2/final_test_evaluation.json](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/improved_cnn_v2_retrained_v2/final_test_evaluation.json):

- sitting F1: 0.4051
- using_phone F1: 0.1987
- waving_hand F1: 0.1868
- jumping F1: 0.0839
- crouching/kicking/pointing/running/walking: F1 0.0
- standing F1: 0.1045

The confusion matrix shows systematic overprediction of waving_hand, while the original high-bias behavior also favored jumping in some classes.

### Checkpoint selection/loading

Checkpoint: [outputs/improved_cnn_v2_retrained_v2/checkpoints/best_model_custom_cnn.pth](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/outputs/improved_cnn_v2_retrained_v2/checkpoints/best_model_custom_cnn.pth).

The checkpoint was selected at epoch 17 on validation Macro F1 and the run ended at epoch 30 rather than epoch 50. The persisted metrics do not reveal whether the stop was intentional early stopping or infrastructure interruption.

### Assessment

Confirmed:

- Additional spatial dropout did not improve validation accuracy.
- The run remained near chance and collapsed onto a small subset of classes.
- The run ended at 30/50 epochs, so the benchmark was not a fully completed 50-epoch V2 run.

Likely causes:

- Inadequate representational capacity or discriminative temporal features.
- SpatialDropout2d may be too aggressive for this already shallow scratch-trained representation.
- Fixed 16-frame mean+max pooling discards the temporal sequence.
- Low or noisy gradient learning is possible but not confirmed without diagnostic passes.

## Confirmed bugs

1. **Model construction includes unused modalities in video-only runs.** `scripts/train.py` instantiates `MultimodalHARModel` with `mode="video_only"`, and video-only forward uses only `video_branch` plus `video_only_classifier`. The sensor branch, fusion classifier, and sensor-only classifier remain in the state/parameter count and optimizer parameter list. This is inefficient but not a demonstrated source of incorrect video predictions.

2. **The benchmark table mixes validation and test checkpoint evidence.** Checkpoints are selected by validation Macro F1, while test accuracy is only reported later. The original/retrained artifact names and sizes are still auditable, but test files must not be used in future checkpoint selection or tuning.

3. **The persisted run manifests do not always record the exact staged-fine-tuning flags.** This prevents a complete reproducibility/parameter-freezing audit from manifest files alone.

4. **The retrained Custom CNN run did not complete all 50 requested epochs.** Its manifest says 50 requested and 30 completed; the benchmark cannot be interpreted as a completed 50-epoch V2 run.

## Possible causes not confirmed

- MobileNet pretrained weights were not loaded due to offline torchvision fallback.
- MobileNet normalization mismatch beyond the established ImageNet normalization.
- The 16-frame uniform sampling missed transient activity phases.
- The mean+max pooling representation lost class-specific temporal ordering.
- The cosine schedule or staged unfreezing LR schedule caused unstable validation loss.
- Class weighting excessively favored minority/majority classes and affected checkpoint ranking.
- Custom CNN capacity is insufficient even before considering temporal ordering.
- SpatialDropout2d and 0.35 projection/head dropout jointly suppress a shallow model.
- Cache reuse changed due to stale cached tensors; cache validation should reject shape and NaN/Inf, but no cache hit/miss audit for these runs is preserved.
- Slow or absent gradient flow in Custom CNN is possible but not established.

## Prioritized correction plan

### Priority 1: Add read-only pipeline validation

1. Print and assert the shape of video, label, logits, and prediction tensors for both branches.
2. Verify every label index equals the index mapping and every class appears in train/validation batches.
3. Count and report total/trainable parameters for the active video-only path separately from unused branches.
4. Record a one-line initialization audit: MobileNet source (`torchvision` or fallback), pretrained weight result, normalization config, freeze state, and optimizer groups.
5. Preserve staged-finetuning flags and exact unfreeze epoch in manifests.

### Priority 2: Screen the Custom CNN without held-out test data

1. Run a forward/backward pass on a 2–8 sample training subset.
2. Verify non-zero gradients for all active Custom CNN layers and the 10-way head.
3. Check whether the model can overfit a very small balanced training subset.
4. If it cannot overfit, treat representational capacity, gradient flow, tensor shape, or label encoding as primary suspects.
5. If it can overfit, tune regularization, augmentation, temporal coverage, and learning-rate schedule before adding capacity.

### Priority 3: Verify and simplify the MobileNetV2 fine-tuning stage

1. Confirm official pretrained weights were loaded and not silently replaced by random/fallback weights.
2. Confirm batch/frame normalization matches the pretrained backbone.
3. Verify the frozen and unfrozen parameter sets at epoch 15.
4. Compare frozen-head-only and staged-unfreeze validation curves.
5. Stop based on validation Macro F1 or validation accuracy, but do not switch based on test metrics.

### Priority 4: Small screening experiments

All experiments should use only train/validation data and the existing subject-independent split.

**Custom CNN**

- Baseline active-video-only parameter setup, single LR, no class weights.
- Same setup with class weights.
- Reduced dropout 0.20–0.25 as a two-run screen.
- Uniform 16-frame baseline with and without temporal mean+max pooling as-is.
- If representational capacity is verified, a modest capacity increase can be screened; introducing a new architecture is not required.

**MobileNetV2**

- Frozen-head baseline for 10–15 epochs.
- One-stage full fine-tune at 1e-5 or 2e-5.
- Staged fine-tune with existing 15-epoch head phase and backbone LR 1e-5.
- 16-frame uniform sampling with and without a lightweight augmentation screen.

Keep each screen to one or two changed factors at a time. Selection should use validation Macro F1 and validation accuracy from the existing validation split only.

## Required regression tests

1. `test_target_class_mapping_round_trip`: verify names map to indices 0–9 and indices map back unchanged.
2. `test_video_tensor_contract`: verify `(B,3,16,112,112)` shape and finite normalized values.
3. `test_uniform_sampling_contract`: verify 16 frames are returned for clips and short clips repeat only the final valid frame.
4. `test_cache_validation_contract`: verify stale, wrong-shape, NaN, and Inf cache entries are rejected and recomputed.
5. `test_custom_cnn_forward_shapes`: verify branch output `(B,128)` and video-only logits `(B,10)`.
6. `test_custom_cnn_gradients`: verify non-zero gradients in convolutional, temporal projection, and classifier parameters on a small batch.
7. `test_custom_cnn_small_subset_overfit`: verify the model can reach near-zero loss on a tiny balanced subset.
8. `test_mobilenet_pretrained_loading`: explicitly report weight source and verify backbone tensors differ from random initialization.
9. `test_mobilenet_classifier_replacement`: verify the 10-way video head receives 128-D embeddings and is trainable.
10. `test_freeze_unfreeze_state`: verify the correct parameter sets require gradients before and after the staged transition.
11. `test_subject_split_isolation`: verify no repeated video, recording, session, or subject across train/validation/test.
12. `test_checkpoint_selection`: verify saved checkpoint corresponds to the best validation Macro F1 and record epoch, accuracy, F1, optimizer, and scheduler state.
13. `test_no_test_usage_during_training`: assert test loader creation/evaluation is absent from training and checkpoint-selection code paths.

## Missing evidence preventing a stronger conclusion

1. The exact original and retrained shell/notebook commands are not preserved in the experiment manifests.
2. The original MobileNet manifest does not explicitly record pretrained source, staged freeze state, or parameter counts.
3. The retrained MobileNet manifest does not record staged-fine-tuning flags even though the LR/GPU memory pattern suggests unfreezing.
4. The Custom CNN checkpoints are present, but the artifacts do not preserve a gradient audit or a small-subset overfit run.
5. Cache hit/miss/corruption statistics are not preserved in the four run summaries.
6. There is no per-experiment loss/accuracy history for every output variant in one consistent schema for all four runs.
7. The retrained Custom CNN run's 30-epoch completion lacks an explicit reason (early stopping versus interruption).
8. The repository does not preserve raw input-frame-level visualizations or clip-duration sampling audits for the four runs.
9. The exact loaded checkpoint architecture arguments are only inferable from model conventions and experiment names; evaluation scripts should be made to emit them directly in a manifest.

## Final recommendation

The smallest evidence-supported next step is **not** to change the architecture or introduce a new architecture. It is to add and run read-only pipeline/gradient/pretrained-loading diagnostics, then perform two narrow validation-only screens:

1. Custom CNN: active-parameter audit plus small-subset overfit test before any architecture or regularization change.
2. MobileNetV2: verify pretrained/freeze/normalization state, then compare the existing staged schedule with a frozen-head-only or one-stage low-LR screen.

No held-out test data should be included in any of these screens.

## Primary evidence links

- [Central class mapping](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/config.py#L19-L38)
- [Video preprocessing](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/data/video_loader.py#L21-L124)
- [Dataset indexing](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/data/indexer.py#L246-L410)
- [Training and checkpoint selection](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/scripts/train.py#L399-L425)
- [Staged fine-tuning transition](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/scripts/train.py#L498-L512)
- [MobileNetV2 branch](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/models/video_branch.py#L254-L334)
- [Original Custom CNN branch](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/models/video_branch.py#L337-L388)
- [Custom CNN V2 branch](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/models/video_branch.py#L391-L447)
- [Video-only and full-model heads](https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct/blob/master/src/models/fusion_model.py#L118-L126)
