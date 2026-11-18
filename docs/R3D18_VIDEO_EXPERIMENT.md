# R3D-18 video-only experiment

The dedicated Kaggle notebook `notebooks/kaggle_r3d18_experiment.ipynb` checks out `improved-models` and runs a separate experiment without changing existing benchmark artifact paths. The implementation uses torchvision's `R3D_18_Weights.KINETICS400_V1`, reverses the existing ImageNet normalization before applying the Kinetics-400 normalization, and raises a clear error if pretrained weight loading fails.

Run from the repository root (Kaggle GPU with internet enabled):

```bash
python scripts/train_r3d18.py \
  --data_root data/genuine_mmact \
  --video_backbone r3d18 \
  --mode video_only \
  --experiment_name r3d18_kinetics400_video_only \
  --epochs 50 --batch_size 4 --lr 0.0001 --weight_decay 0.0001 \
  --dropout 0.35 --seed 42 \
  --output_dir outputs/video_model_improvements/r3d18 \
  --cache_dir outputs/cache --num_workers 2
```

The trainer checkpoints after every epoch and resumes automatically from `checkpoints/latest_model_r3d18.pth`. Validation macro-F1 chooses the best checkpoint. Only after training is complete is that checkpoint evaluated on test; the run refuses to start unless the existing subject-independent split yields exactly 852 test samples, and it checks the IDs again before writing predictions. It records split IDs, counts, seed, GPU, hyperparameters, weight identifier, and completed epoch in `experiment_manifest.json`.

Outputs: `checkpoints/best_model_r3d18.pth`, `checkpoints/latest_model_r3d18.pth`, `epoch_history.json` / `.csv`, `metrics.json`, `predictions.json`, and `experiment_manifest.json`.
