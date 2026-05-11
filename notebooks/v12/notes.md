# v12 Notes

v12 builds on v11 (1051.1, new best) with four heuristic fixes and a fresh model trained on the 2500-game both-sides dataset already in `data/20260510_141652/`.

## Files

- `train_v12_ranker.py` — v12 trainer (same architecture as v9, v12 artifact paths).
- `v12_training_policy.ipynb` — self-contained Kaggle/Colab notebook; asks for `HF_TOKEN`, embeds trainer, uses local 2500-game CSV by default (or downloads newest from HF), uploads to `devaanshpa/orbit-wars-agent/v12`.

## Heuristic Changes (main.py)

### Proactive defense tightened
`enemy_ships * 0.18` → `enemy_ships * 0.30` in `_build_policy`. A 100-ship threat 10 turns away now adds 30 reserve ships instead of 18, preventing overcommit attacks that leave planets exposed.

### Endgame reserve ramp
`reserve_for()` now linearly releases up to 55% of planet reserves over the 80 turns before `ENDGAME_STEP` (steps 325–405). Previously reserves dropped to near-zero abruptly at step 405; the ramp frees ships progressively for a stronger pre-endgame push.

### One-ply opponent modeling
`_planner_projected_value()` now checks each source planet in a candidate's parts: if the enemy can capture that planet before our fleet lands (garrison + production accrual < enemy fleet size), the candidate is penalized by `source.production * remaining_turns * 1.2`. This prevents attacks that overcommit ships and leave the source planet capturable.

### Production-race aggression
Strengthened the production-race escalation from v11:
- Triggers earlier: step 80 (was 100)
- Wider threshold: ratio < 0.92 (was 0.85) — any meaningful production deficit engages the mode
- Steeper ramp: multiplier 3.0 (was 2.0)
- Higher max drawdown: 60% (was 35%)
- Floor: reserve[pid] kept ≥ 1 per planet

At ratio 0.82 (10% behind): draws 30% of reserves into attack budget. At ratio 0.72 (20%+ behind): draws the full 60%. Losing the production race while sitting on ships is a slow death — this forces expansion over passive defense.

## Training Defaults

Same as v9 — these were well-tuned on the v7 dataset and should transfer:

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
- Checkpoint every: 40 epochs → `v12/checkpoints/` on HF

## Dataset

Local: `data/20260510_141652/candidates_v7.csv` (~312k rows, 2500 games, both sides).

The notebook has this path commented out in cell 3 for easy override. By default it downloads the newest HF `data/*/candidates_v7.csv` (same file).

## Run Order

1. Open `v12_training_policy.ipynb` on Kaggle (GPU T4 x2) or Colab.
2. Enter `HF_TOKEN` when prompted (cell 1).
3. Optionally uncomment the local CSV path in cell 3 if running locally.
4. Run all cells. Checkpoints upload every 40 epochs; final artifacts go to `v12/`.
5. Download `model_weights_v12.json` to local `models/`.
6. Build submission:
   ```bash
   python build_submission.py --weights models/model_weights_v12.json --output models/v12_kaggle/main.py
   ```
7. Submit:
   ```bash
   kaggle competitions submit orbit-wars -f models/v12_kaggle/main.py -m "v12 tighter defense + opponent modeling + retrained ensemble"
   ```
