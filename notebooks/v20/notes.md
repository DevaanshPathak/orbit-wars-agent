# v20 Notes

v20 is a conservative counterfactual RL workbench. The goal is not to chase the
offline GRPO reward at all costs; it is to keep the strong SFT/heuristic policy
intact and only accept GRPO updates that pass validation safety gates.

## What v20 Changes

- Adds `rl-counterfactual-v20` to `generate_training_data.py`.
- Writes `candidates_v20.csv` instead of overwriting older v7/v19 data.
- Uses longer default rollouts in v20 mode: 35 turns and 10 candidate rollouts
  per turn when the user does not override them.
- Promotes unselected candidates with positive rollout deltas into direct
  counterfactual positives instead of treating them as weak unknowns.
- Downlabels selected candidates when their rollout crashes or causes a large
  negative margin delta.
- Keeps SFT as the main policy and constrains GRPO with validation top-1 and KL
  gates.

## Files

- `sft_training_policy.ipynb`: self-contained Kaggle/Colab SFT notebook. It reads
  `HF_TOKEN`, downloads training data from Hugging Face, logs every epoch, saves
  checkpoints/graphs, and uploads artifacts to `v20/sft/`.
- `grpo_training_policy.ipynb`: self-contained constrained GRPO notebook. It
  downloads the v20 SFT artifact, applies top-1 and KL safety gates, caps model
  blend, carries SFT ensemble members into the final artifact, and uploads to
  `v20/grpo/`.
- `train_sft_policy.py`: optional local CLI copy of the SFT trainer.
- `train_grpo_policy.py`: optional local CLI copy of the GRPO trainer.

## Default Data

Until a dedicated v20 dataset is generated, both notebooks default to the known
uploaded dataset:

```text
devaanshpa/orbit-wars-agent/data/20260520_061012/candidates_v19.csv
```

That keeps v20 runnable immediately. For the intended v20 run, generate a fresh
dataset and then set the notebook data remote path to the uploaded v20 CSV:

```powershell
python generate_training_data.py --mode rl-counterfactual-v20 --games 2500 --workers 16 --max-candidates-per-turn 48
```

The output path will be:

```text
data/<run_start_timestamp>/candidates_v20.csv
```

Upload that folder to Hugging Face as `data/<run_start_timestamp>/`, then set:

```text
v20_SFT_DATA_REMOTE_PATH=data/<run_start_timestamp>/candidates_v20.csv
v20_GRPO_DATA_REMOTE_PATH=data/<run_start_timestamp>/candidates_v20.csv
```

## GRPO Safety Gates

v19 showed classic reward over-optimization: reward gap improved while
validation top-1 collapsed. v20 prevents that by:

- Starting best state from the SFT baseline.
- Rejecting epochs whose validation top-1 drops by more than 0.008 from SFT.
- Rejecting epochs whose average KL exceeds 0.42.
- Penalizing top-1 drop and KL excess in the objective.
- Tuning final model blend only up to a conservative cap of 0.24.
- Carrying two SFT ensemble members into the final GRPO artifact.

## Run Order

1. Generate or reuse data.
2. Add Kaggle Secret `HF_TOKEN`.
3. Run `sft_training_policy.ipynb` on Kaggle 2*T4.
4. Confirm `v20/sft/model_weights_v20_sft.json` exists on Hugging Face.
5. Run `grpo_training_policy.ipynb` on Kaggle 2*T4.
6. Confirm `v20/grpo/model_weights_v20_grpo.json` exists on Hugging Face.
7. Pull the GRPO artifact locally into `models/`.
8. Build and test the submission with `build_submission.py`.

## Defaults

SFT:

- 4-member ensemble
- 180 epochs
- 256 candidate groups per batch
- LR 0.00055
- dropout 0.12
- checkpoints every 30 epochs

GRPO:

- 60 epochs
- 192 candidate groups per batch
- 12 samples per group
- temperature 0.95
- LR 0.00010
- KL weight 0.35
- entropy weight 0.030
- supervised anchor 0.32
- patience 14
- top-1 drop tolerance 0.008
- max KL 0.42
- blend cap 0.24
- carried SFT members 2
