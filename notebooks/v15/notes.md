# v15 Notes

v15 trains a fresh 8-member ensemble ranker on top of the v13 heuristic (same base as v14) using the **pinned** `data/20260516_032302` dataset (5k games × 2 sides = 10k game-plays, 608k rows, generated 2026-05-16).

## Files

- `train_v15_ranker.py` — v15 trainer (same architecture as v9/v12/v14; artifacts to `v15/` on HF).
- `v15_training_policy.ipynb` — self-contained Kaggle/Colab notebook; asks for `HF_TOKEN`, fetches the **pinned** `data/20260516_032302/candidates_v7.csv` from HF (no auto-discovery), uploads to `devaanshpa/orbit-wars-agent/v15`.

## Dataset

Pinned to `data/20260516_032302/candidates_v7.csv`:
- 5000 seeds × 2 sides = 10,000 game-plays
- 608,298 rows (423,669 positive)
- Opponents: random / nearest / starter / greedy / rusher (both sides)
- Generated 2026-05-16 with v13 heuristic agent

## Heuristic Baseline

v13 heuristic (see `notebooks/v13/notes.md`):
- Staging re-enabled with contested-front guard (enemy ETA ≤ 15)
- Beam width: 5
- Per-planet production-race drawdown (safe planets only, enemy ETA > 20)
- Opportunity scoring for low-garrison enemy planets (+production × 8.0)
- Comet ETA cap extended to 22 when leading in production

## Training Defaults

Same as v12/v13/v14 (well-tuned):

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
- Checkpoint every: 40 epochs → `v15/checkpoints/` on HF

## Run Order

1. Open `v15_training_policy.ipynb` on Kaggle (GPU T4 x2) or Colab.
2. Enter `HF_TOKEN` when prompted (cell 1).
3. Run all cells. Cell 3 downloads the pinned `data/20260516_032302/candidates_v7.csv` directly.
4. Checkpoints upload every 40 epochs; final artifacts go to `v15/`.
5. Download `model_weights_v15.json` to local `models/`.
6. Build submission:
   ```bash
   python build_submission.py --weights models/model_weights_v15.json --output models/v15_kaggle/main.py
   ```
7. Submit:
   ```bash
   kaggle competitions submit orbit-wars -f models/v15_kaggle/main.py -m "v15 v13 heuristic + retrained ensemble on 5k-game dataset (20260516_032302)"
   ```
