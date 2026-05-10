# v9 Notes

v9 is the first serious SFT + GRPO scale-up after the v8 smoke pipeline. It is designed for Kaggle TPU v5e-8.

## Files

- `sft_tpu_training_policy.ipynb` is self-contained for Kaggle TPU. It writes the embedded v9 trainer at runtime, asks for `HF_TOKEN`, spawns SFT ensemble members across 8 XLA processes, uploads checkpoints every 30 epochs, and uploads final artifacts to `devaanshpa/orbit-wars-agent/v9/sft`.
- `grpo_tpu_training_policy.ipynb` is self-contained for Kaggle TPU. It downloads the v9 SFT artifact, spawns GRPO-style reward-tuning members across 8 XLA processes, uploads checkpoints every 30 epochs, and uploads final artifacts to `devaanshpa/orbit-wars-agent/v9/grpo`.
- `train_v9_tpu.py` is the optional local/CLI copy of the same trainer embedded in both notebooks.

## TPU Design

v9 uses member parallelism rather than synchronized distributed gradient descent. Each TPU core trains one independent compact MLP member from a different seed, then the main process aggregates the member JSON files into the same ensemble format `main.py` already supports. This keeps TPU use simple and avoids fragile all-reduce behavior for the custom candidate-ranking losses.

The v9 trainer uses fixed-size row and pair batches instead of v8's dynamic candidate-group training loop. This is intentional because dynamic group shapes can cause repeated XLA recompilation on TPU.

## Dataset Target

- 1000 games with both sides enabled.
- Roughly 2000 logged game perspectives.
- Use the full baseline and prior-version mix:
  - random
  - nearest
  - starter
  - greedy
  - rusher
  - v4/v5/v6/v7/v8 artifacts when available
- Keep generated data under `data/<run_start_timestamp>/`.
- Upload the finished run to Hugging Face under `data/<run_start_timestamp>/`.

## Goal

v9 should prove that the constrained RL policy can beat the best heuristic baseline after seeing enough data to reduce overfitting.

Expected target:

- Beat v4/v5 locally on held-out seeds.
- Improve over v8 SFT-only.
- Show GRPO improvement without increasing invalid actions, sun losses, or timeout risk.
- Public score target range: 1250-1450 if the v8 pipeline is stable.

## SFT Direction

- Keep HF `candidates_v7.csv` or successor CSV as the default input.
- Increase ensemble size if training time allows.
- Track per-phase metrics, not only overall top1.
- Watch for overfitting on selected heuristic actions.

Suggested starting shape:

- SFT epochs: 160-220
- Row batch size: 4096
- Pair batch size: 4096
- Ensemble size: 8 on TPU v5e-8
- Dropout: 0.12-0.18
- Patience: 24-30

## GRPO Direction

For v9, GRPO should still be conservative.

- Keep KL anchoring to SFT.
- Keep supervised anchor.
- Prefer stable improvement over aggressive reward chasing.
- Validate against held-out seeds and prior versions before submitting.

Suggested starting shape:

- GRPO epochs: 100-140
- Row batch size: 4096
- Pair batch size: 4096
- Members: 8 on TPU v5e-8
- KL weight: 0.055-0.075
- Supervised anchor: 0.10-0.16
- Patience: 20-28

## Run Order

1. Run `sft_tpu_training_policy.ipynb` on Kaggle TPU v5e-8.
2. Confirm `v9/sft/model_weights_v9_sft.json` exists in Hugging Face.
3. Run `grpo_tpu_training_policy.ipynb` on Kaggle TPU v5e-8.
4. Fetch `v9/grpo/model_weights_v9_grpo.json` into local gitignored `models/`.
5. Build and test a Kaggle submission with `build_submission.py`.

## Go / No-Go

Move beyond v9 only if:

- v9 beats v8 in local tournaments.
- v9 beats v4/v5 by a visible margin.
- Validation top1/rank improves and local gameplay improves.
- Kaggle public score moves meaningfully above the current heuristic ceiling.
