# v19 Notes

v19 is a counterfactual RL workbench that replaces static label-based rewards with **shallow rollout deltas** — causal signals that measure the actual game-state impact of individual candidate moves.

## Architecture Shift from v18

v18 trained on correlational signals: `future_advantage_delta_15`, `game_result`, and `label`. These measured what happened *after* the heuristic chose, but didn't isolate individual candidate effects. v19 solves this by running shallow counterfactual rollouts during data generation.

For each turn, the top N candidates (default 6) are projected forward 25 turns using a ledger-based simulator. The resulting game-state delta vs. the heuristic-selected baseline gives 5 new causal fields:

- `cf_margin_delta` — ship advantage delta vs baseline
- `cf_prod_delta` — production gap delta vs baseline
- `cf_planet_delta` — planet count delta vs baseline
- `cf_survival` — 1.0 if agent still alive at rollout end
- `cf_crash` — 1.0 if fleet hit the sun during rollout

## Files

- `sft_training_policy.ipynb` — self-contained Kaggle/Colab SFT notebook. Reads `HF_TOKEN` from Kaggle Secrets, downloads `candidates_v19.csv` from HF, trains with counterfactual-weighted cross-entropy, and uploads artifacts to `v19/sft/`.
- `grpo_training_policy.ipynb` — self-contained Kaggle/Colab GRPO notebook. Downloads `candidates_v19.csv` + v19 SFT artifact from HF, trains with rollout-delta reward, and uploads to `v19/grpo/`.
- `train_sft_policy.py` — optional local CLI copy of the SFT trainer.
- `train_grpo_policy.py` — optional local CLI copy of the GRPO trainer.
- `notes.md` — this file.

## Data

Default data source:

```text
devaanshpa/orbit-wars-agent/data/20260520_061012/candidates_v19.csv
```

Generate with:

```bash
python generate_training_data.py --mode rl-counterfactual --games 500 --rollout-turns 25 --rollout-candidates 6
```

Override with:

```bash
set CANDIDATES_CSV=path\to\candidates_v19.csv
```

## Key Differences from v18

### SFT
- **Counterfactual-weighted loss**: cross-entropy weighted by `cf_margin_delta` so genuinely good candidates get higher training weight
- **Same architecture**: 4-member ensemble MLP, 180 epochs

### GRPO
- **Rollout-delta reward**: `cf_margin_delta` is the dominant reward signal (~40% of total variance), replacing `label_signal`
- **Reduced label weight**: `label_signal` weight cut from 1.15 to 0.40
- **Shorter training**: 50 epochs (was 160) due to higher signal per epoch
- **Higher LR**: 0.00035 (was 0.00018)
- **Tighter KL**: 0.15 (was 0.10)

### Data Generation
- **Experimental candidates**: comet denial, split reserves, delayed attacks added to candidate pool
- **Shallow rollout simulator**: ledger-based projection (~2-5ms per candidate, not full env simulation)
- **New CSV schema**: 5 additional `cf_*` columns

## Run Order

1. Generate data: `python generate_training_data.py --mode rl-counterfactual --games 500 --no-upload`
2. Upload data to HF path `data/20260520_061012/`
3. Add Kaggle Secret `HF_TOKEN`
4. Run `sft_training_policy.ipynb`
5. Confirm upload of `v19/sft/model_weights_v19_sft.json`
6. Run `grpo_training_policy.ipynb`
7. Confirm upload of `v19/grpo/model_weights_v19_grpo.json`
8. Pull GRPO model locally
9. Build: `python build_submission.py --weights models/v19/model_weights_v19_grpo.json --output models/v19_kaggle/main.py`
10. Test locally, then submit

## Defaults

SFT:

- 4-member ensemble
- 180 epochs
- 256 candidate groups per batch
- LR 0.00055
- dropout 0.12
- counterfactual weight scaling: 0.6 + clamp(cf_margin_delta / 60, -0.4, 0.8)

GRPO:

- 50 epochs
- 192 candidate groups per batch
- 12 samples per group
- temperature 0.85
- LR 0.00035
- KL weight 0.15
- entropy weight 0.018
- supervised anchor 0.18

Use Kaggle 2×T4 for SFT and GRPO.
