# v16 Notes

v16 trains a fresh 8-member ensemble ranker on top of the v13 heuristic using the **pinned** `data/20260517_074915` dataset (5k games × 2 sides = 10k game-plays, 608k rows, generated 2026-05-17).

## Files

- `train_v16_ranker.py` — v16 trainer (same architecture as v9/v12/v14/v15; artifacts to `v16/` on HF).
- `v16_training_policy.ipynb` — self-contained Kaggle/Colab notebook; asks for `HF_TOKEN`, fetches the **pinned** `data/20260517_074915/candidates_v7.csv` from HF (no auto-discovery), uploads to `devaanshpa/orbit-wars-agent/v16`.

## Dataset

Pinned to `data/20260517_074915/candidates_v7.csv`:
- 5000 seeds x 2 sides = 10,000 game-plays
- 608,172 rows (422,748 positive)
- Opponents: random / nearest / starter / greedy / rusher (both sides)
- Generated 2026-05-17 with v13 heuristic agent

## Training Defaults

Same as v15 (well-tuned):

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
- Checkpoint every: 40 epochs

## Run Order

1. Upload the dataset to HF first (one-time, from local machine) — see notes below.
2. Open `v16_training_policy.ipynb` on Kaggle (GPU T4 x2) or Colab.
3. Enter `HF_TOKEN` when prompted (cell 1).
4. Run all cells. Cell 3 downloads the pinned dataset directly.
5. Checkpoints upload every 40 epochs; final artifacts go to `v16/`.
6. Download `model_weights_v16.json` to local `models/`.
7. Build submission: `python build_submission.py --weights models/model_weights_v16.json --output models/v16_kaggle/main.py`
8. Submit: `kaggle competitions submit orbit-wars -f models/v16_kaggle/main.py -m "v16 ensemble on 20260517_074915"`
