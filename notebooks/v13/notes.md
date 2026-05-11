# v13 Notes

v13 builds on v12 (1049.8, new best) with five heuristic improvements and a fresh model trained on the 5k-game both-sides dataset generated from the v12 heuristic.

## Files

- `train_v13_ranker.py` — v13 trainer (same architecture as v9/v12, v13 artifact paths).
- `v13_training_policy.ipynb` — self-contained Kaggle/Colab notebook; asks for `HF_TOKEN`, embeds trainer, auto-discovers the newest `data/*/candidates_v7.csv` from HF (picks up the 5k-game dataset once uploaded), uploads to `devaanshpa/orbit-wars-agent/v13`.

## Heuristic Changes (main.py)

### Staging re-enabled
`USE_STAGING = True`. Ships on interior planets were completely idle. The staging pass now runs after the deep planner (one move per turn) and pushes available ships from rear planets toward the nearest safe front planet. Added a guard: if the front planet's `enemy_reach` ETA ≤ 15 turns, staging is skipped so ships aren't moved into a contested planet.

### Wider beam search
`PLANNER_BEAM = 4` → `5`. Gives the beam search 25% more candidate bundles to keep per layer, potentially finding better multi-target combinations within the same time budget.

### Per-planet production-race drawdown
Previously the production-race aggression block drew from ALL planets uniformly. Now planets with `enemy_reach` ETA ≤ 20 are skipped, keeping their defense reserves intact. Ships are only freed from safe interior planets, not from front-line planets that need their garrison.

### Opportunity scoring for low-garrison enemy planets
Added +`production * 8.0` bonus in `_candidate_score` when an enemy planet has `ships < production * 3.5`. This prioritizes striking planets that recently sent fleets out before their garrison recovers.

### Comet ETA extension when ahead
Comet ETA cap raised 18 → 22 turns when `my_production >= enemy_production`. When leading, contesting farther comets is worthwhile. When behind, the tighter cap remains so comets don't drain attack budget.

## Training Defaults

Same as v12 (well-tuned):

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
- Checkpoint every: 40 epochs → `v13/checkpoints/` on HF

## Dataset

The notebook auto-discovers the newest `data/*/candidates_v7.csv` from HF — once the 5k-game dataset (generated locally with `python generate_training_data.py --games 5000 --both-sides --workers 16`) is uploaded, it will be used automatically.

## Run Order

1. Finish 5k-game data generation locally; it uploads automatically to HF under `data/<timestamp>/`.
2. Open `v13_training_policy.ipynb` on Kaggle (GPU T4 x2) or Colab.
3. Enter `HF_TOKEN` when prompted (cell 1).
4. Run all cells. Cell 3 auto-downloads the newest HF dataset. Checkpoints upload every 40 epochs; final artifacts go to `v13/`.
5. Download `model_weights_v13.json` to local `models/`.
6. Build submission:
   ```bash
   python build_submission.py --weights models/model_weights_v13.json --output models/v13_kaggle/main.py
   ```
7. Submit:
   ```bash
   kaggle competitions submit orbit-wars -f models/v13_kaggle/main.py -m "v13 staging + wider beam + safe production-race + opportunity scoring + comet extension"
   ```
