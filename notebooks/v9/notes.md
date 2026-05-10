# v9 Notes

v9 replaces the experimental TPU SFT/GRPO path with a scaled version of the supervised counterfactual ensemble ranker that produced the strongest public score so far.

## Files

- `train_v9_ranker.py` trains a Kaggle-compatible JSON ensemble from `candidates_v7.csv`.
- `v9_training_policy.ipynb` is self-contained for Kaggle/Colab. It embeds the trainer code in the notebook, asks for `HF_TOKEN`, streams epoch logs, downloads the newest Hugging Face `data/*/candidates_v7.csv` by default, and uploads final artifacts to `devaanshpa/orbit-wars-agent/v9`.

## Dataset Target

- 2500 generated games with both sides enabled.
- Roughly 5000 game perspectives.
- Use `random nearest starter greedy rusher` unless running an explicit ablation.
- Keep generated data under `data/<run_start_timestamp>/` and upload to Hugging Face under the same `data/<run_start_timestamp>/` folder.

## Training Defaults

- Ensemble size: 8
- Epochs: 280
- Batch size: 4096
- Pair loss weight: 1.05
- Max pairs per turn: 12
- Learning rate: 0.00045
- Weight decay: 0.00030
- Dropout: 0.13
- Patience: 36
- Score scale: 205
- Checkpoint every: 40 epochs (uploaded to `v9/checkpoints/` when `--upload` is set; override with `V9_CHECKPOINT_EVERY` or `--checkpoint-every`, set to 0 to disable)

## Why Not GRPO

The v8 GRPO smoke submission scored below v7, so v9 returns to the training path that actually improved the leaderboard: supervised counterfactual ranking with a larger dataset and stronger ensemble regularization.

## Run Order

1. Generate the 2500-game both-sides dataset.
2. Run `v9_training_policy.ipynb`.
3. Fetch `v9/model_weights_v9.json` into gitignored `models/`.
4. Build a submission with `build_submission.py`.
5. Validate locally before submitting.

Generated exports, checkpoints, datasets, and submission bundles must stay out of git.
