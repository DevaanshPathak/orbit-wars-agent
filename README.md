# Orbit Wars v2 Heuristic Agent

This repository contains a single-file Orbit Wars Kaggle agent in `main.py`.

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

## Submit

```bash
kaggle competitions submit orbit-wars -f main.py -m "v2 collision race heuristic"
```

## What v2 does

v2 is still pure stdlib heuristics. It keeps the v1 arrival ledger and candidate scaffold, then adds engine-faithful swept collision validation, opponent race pressure, safer opening expansion, comet spawn handling, coordinated attack timing, doomed-planet evacuation, and score-aware endgame behavior.

## What v3 should add

v3 should add replay-driven tuning, a stronger opponent launch model, scripted openings by map class, and a compact forward evaluator for choosing between mutually exclusive capture plans.
