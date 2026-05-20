# v18 Notes

v18 is a reward-resistant constrained RL workbench. It keeps Orbit Wars legality inside the existing candidate action space, then trains:

1. SFT warm start over per-turn candidate groups.
2. GRPO-style policy improvement over the same legal candidate groups.

This is intentionally not raw continuous action RL. Raw angle/ship generation would spend most compute relearning legality, sun avoidance, intercept timing, and fleet sizing. v18 lets RL learn when to deviate from the heuristic without allowing illegal or obviously unsafe actions.

## Files

- `sft_training_policy.ipynb` - self-contained Kaggle/Colab SFT notebook. It reads `HF_TOKEN` from Kaggle Secrets or the environment, downloads the pinned dataset from HF, trains, logs every epoch, uses `torch.nn.DataParallel` on 2*T4 by default, checkpoints, graphs, and uploads artifacts to `v18/sft/`.
- `grpo_training_policy.ipynb` - self-contained Kaggle/Colab GRPO notebook. It reads `HF_TOKEN` from Kaggle Secrets or the environment, downloads the pinned dataset plus `v18/sft/model_weights_v18_sft.json`, trains, logs every epoch, checkpoints, graphs, and uploads artifacts to `v18/grpo/`.
- `train_sft_policy.py` - optional local CLI copy of the SFT trainer.
- `train_grpo_policy.py` - optional local CLI copy of the GRPO trainer.
- `notes.md` - this file.

## Data

Default data source:

```text
devaanshpa/orbit-wars-agent/data/20260517_074915/candidates_v7.csv
```

Override with:

```bash
set CANDIDATES_CSV=path\to\candidates_v7.csv
```

or in the notebook by changing `V18_SFT_DATA_REMOTE_PATH` / `V18_GRPO_DATA_REMOTE_PATH`.

## Reward-Hack Resistance

GRPO uses a bounded multi-component reward:

- label signal, but with reduced weight
- future advantage deltas at 15 and 30 turns
- production and planet-count deltas
- final result and final margin with small clipped weights
- small selected-action anchor
- counterfactual-positive bonus
- explicit penalties for overcommit, missed tactical moves, missed comets, slow expansion, crash candidates, excessive ship cost, and very late ETA
- KL penalty to the SFT reference policy
- supervised anchor distribution so GRPO cannot drift into arbitrary high-reward artifacts
- validation objective includes top-1, rank fraction, KL drift, negative reward gap, and failure exposure

This does not make reward hacking impossible, but it makes obvious single-signal hacks much less likely to pass validation.

## Run Order

1. Add a Kaggle notebook Secret named `HF_TOKEN` and enable notebook access to it.
2. Run `sft_training_policy.ipynb`.
3. Confirm it uploaded `v18/sft/model_weights_v18_sft.json` to HF.
4. Run `grpo_training_policy.ipynb`.
5. Confirm it uploaded `v18/grpo/model_weights_v18_grpo.json` to HF.
6. Pull the GRPO model locally into `models/`.
7. Build:

```bash
python build_submission.py --weights models/v18/model_weights_v18_grpo.json --output models/v18_kaggle/main.py
```

8. Test locally, then submit:

```bash
kaggle competitions submit orbit-wars -f models/v18_kaggle/main.py -m "v18 constrained SFT+GRPO"
```

## Defaults

SFT:

- 4-member ensemble
- 220 epochs
- 256 candidate groups per batch
- multi-GPU DataParallel enabled by default with `V18_SFT_MULTI_GPU=1`
- LR 0.00055
- dropout 0.12
- rank weight 0.55
- checkpoints every 30 epochs

GRPO:

- 160 epochs
- 192 candidate groups per batch
- 12 samples per group
- temperature 0.85
- LR 0.00018
- KL weight 0.10
- entropy weight 0.018
- supervised anchor 0.22
- checkpoints every 30 epochs

Use Kaggle 2*T4 for SFT and GRPO. Kaggle's current PyTorch build does not support P100's `sm_60` CUDA target, so P100 fails before training starts. CPU works for smoke tests only.
