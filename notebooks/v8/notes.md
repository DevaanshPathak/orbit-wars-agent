# v8 Notes

v8 is the first reinforcement-learning-oriented version. It keeps the current Orbit Wars heuristic candidate generator as the legal action space and trains policies to rank those candidates.

## Files

- `sft_training_policy.ipynb` is self-contained for Kaggle: it downloads `candidates_v7.csv` from Hugging Face with `HF_TOKEN`, trains SFT directly in notebook cells, uploads artifacts to `devaanshpa/orbit-wars-agent/v7/sft`, and displays saved metrics/graphs.
- `grpo_training_policy.ipynb` is self-contained for Kaggle: it downloads `candidates_v7.csv` and the SFT artifact from Hugging Face with `HF_TOKEN`, trains constrained GRPO directly in notebook cells, uploads artifacts to `devaanshpa/orbit-wars-agent/v7/grpo`, and displays saved metrics/graphs.
- `train_sft_policy.py` is an optional local CLI copy of the SFT trainer.
- `train_grpo_policy.py` is an optional local CLI copy of the GRPO trainer.

## Artifact Rules

Generated model artifacts, predictions, and graphs stay under the notebook export folder, which is gitignored locally. On Kaggle the self-contained notebooks default to `/kaggle/working/v8_exports/...`. Do not commit trained weights or generated submissions.

The current remote artifact folders intentionally use `v7/sft` and `v7/grpo` because that was the requested Hugging Face layout for this training cycle.

## Training Order

1. Finish and upload the v7 candidate CSV to Hugging Face under `data/<run_start_timestamp>/candidates_v7.csv`.
2. Run SFT.
3. Run GRPO from the Hugging Face candidate CSV and the SFT artifact at `v7/sft/model_weights_v8_sft.json`.
4. Fetch the final GRPO JSON into `models/`.
5. Build a local submission with `build_submission.py`.
6. Validate locally against v4/v5/v6/v7 before Kaggle submission.

## Notebook Defaults

The notebooks are set for a 1000-game both-sides dataset on Kaggle 2*T4:

- SFT: 180 epochs, 192 candidate groups per batch, 3 ensemble members, dropout 0.14, patience 28.
- GRPO: 120 epochs, 160 candidate groups per batch, 10 samples per group, temperature 0.90, KL weight 0.065, supervised anchor 0.14, patience 24.

These values can still be overridden with environment variables before the training cell.

## Important Caveat

This v8 code uses constrained candidate-policy RL, not raw continuous angle/ship RL. That is intentional: raw actions are too brittle for Orbit Wars and would waste most training on invalid or unsafe moves.
