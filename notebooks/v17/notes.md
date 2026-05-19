# v17 Notes

v17 trains an 8-member ensemble ranker on top of the v16 heuristic using the pinned
`data/20260517_074915/candidates_v7.csv` dataset. The dataset has been uploaded to
Hugging Face at `devaanshpa/orbit-wars-agent/data/20260517_074915/`.

## Files

- `train_v17_ranker.py` - v17 trainer; defaults to the pinned HF dataset and uploads artifacts to `v17/` on HF.
- `v17_training_policy.ipynb` - self-contained Kaggle/Colab notebook; asks for `HF_TOKEN`, downloads the pinned dataset, trains, checkpoints, graphs, and uploads artifacts to `devaanshpa/orbit-wars-agent/v17`.
- `notes.md` - this file.

## Dataset

Pinned dataset:

- Remote path: `data/20260517_074915/candidates_v7.csv`
- 5000 seeds x 2 sides = 10,000 game-plays
- 608,172 candidate rows
- 422,748 positive rows
- Opponents: random / nearest / starter / greedy / rusher
- Generated with the counterfactual teacher data path and uploaded from local offline output

The trainer still accepts a local override:

```bash
python notebooks/v17/train_v17_ranker.py --csv data/20260517_074915/candidates_v7.csv --upload
```

Without `--csv`, it uses the pinned HF dataset.

## Model Architecture

256 -> 128 -> 64 -> 1 MLP, ReLU activations, dropout, and an 8-member ensemble.

## Training Defaults

- Ensemble size: 8
- Epochs: 300
- Batch size: 4096
- Pair loss weight: 1.10
- Max pairs per turn: 14
- Learning rate: 0.00045
- Weight decay: 0.00030
- Dropout: 0.12
- Patience: 40
- Score scale: 205
- Checkpoint every: 40 epochs to `v17/checkpoints/` on HF

## Run Order

1. Open `v17_training_policy.ipynb` on Kaggle or Colab.
2. Run all cells.
3. Enter `HF_TOKEN` when prompted.
4. Cell 3 downloads the pinned dataset from HF.
5. Training logs print every epoch.
6. Checkpoints upload every 40 epochs; final artifacts and graphs upload to `v17/`.
7. After training, download `model_weights_v17.json` into local `models/`.
8. Build submission:

```bash
python build_submission.py --weights models/model_weights_v17.json --output models/v17_kaggle/main.py
```

9. Submit:

```bash
kaggle competitions submit orbit-wars -f models/v17_kaggle/main.py -m "v17 ensemble on 20260517_074915"
```
