# Orbit Wars v5 Heuristic Agent

This repository contains a single-file Orbit Wars Kaggle agent in `main.py`, plus v5 notebooks for offline analysis and training.

## Run locally

Install the environment:

```bash
pip install "kaggle-environments>=1.28.0"
```

Run one sanity-check game:

```bash
python main.py
```

Run local benchmarks:

```bash
python test_local.py
python test_local.py --games 10 --baselines random nearest starter greedy rusher --four-player
python test_local.py --games 50 --compare-git-ref 8f5b855
```

## Generate v5 training data

The generator runs local games, logs candidate features at decision time, writes one folder for the whole run under gitignored `data/<run_start_timestamp>/`, and uploads that run folder to Hugging Face under `devaanshpa/orbit-wars-agent/data/<run_start_timestamp>`.

```bash
python generate_training_data.py --games 100 --opponents random nearest starter
```

Each game is identified inside the CSV by `game_id`; games do not create separate timestamped folders.

For more visible progress and better CPU use on a multi-core machine:

```bash
python generate_training_data.py --games 100 --opponents random nearest starter --workers 4
```

For a local smoke test without upload:

```bash
python generate_training_data.py --games 1 --max-rows 200 --no-upload
```

The generator does not need a GPU. Orbit Wars game simulation and candidate extraction are CPU-bound Python work. Low total CPU usage usually means a single Python worker is saturating one core; use `--workers` to run independent games in parallel.

Kaggle may print optional OpenSpiel environment warnings in some installs. The generator suppresses that import noise by default; pass `--show-env-imports` if you need to debug Kaggle environment loading.

## Submit

```bash
kaggle competitions submit orbit-wars -f main.py -m "v5 model-guided deep planner"
```

## Hugging Face artifacts

Training notebooks write generated outputs under local `notebooks/**/exports/`, `models/`, and `data/` folders. These paths are gitignored. The v5 training notebook trains, evaluates, saves graphs, and uploads all exported artifacts to Hugging Face repo `devaanshpa/orbit-wars-agent` under the remote `v5/` folder after prompting for `HF_TOKEN`.

Later local model downloads should go into the root `models/` folder, which is also gitignored.

Do not commit trained models, checkpoints, replay datasets, generated model exports, or submission bundles to GitHub.

## What v5 does

v5 keeps the v4 guarded heuristic core, then adds model-ready candidate features and a bounded deep planner over generated expansion, attack, and comet candidates. If no local model weights are available, the agent stays Kaggle-safe and falls back to heuristic scoring.

## What v6 should add

v6 should consume trained v5 artifact metrics, add replay-backed candidate labels, and tune the planner/model blend against leaderboard and local head-to-head results.
