# v9 Notes

v9 is the first serious SFT + GRPO scale-up after the v8 smoke pipeline.

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
- Batch groups: 192-256
- Ensemble size: 3-4
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
- Batch groups: 128-192
- Samples per group: 8-12
- KL weight: 0.055-0.075
- Supervised anchor: 0.10-0.16
- Patience: 20-28

## Go / No-Go

Move beyond v9 only if:

- v9 beats v8 in local tournaments.
- v9 beats v4/v5 by a visible margin.
- Validation top1/rank improves and local gameplay improves.
- Kaggle public score moves meaningfully above the current heuristic ceiling.
