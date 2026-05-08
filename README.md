# Orbit Wars v1 Heuristic Agent

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
python test_local.py --games 10 --baselines random nearest starter --four-player
```

## Submit

```bash
kaggle competitions submit orbit-wars -f main.py -m "v1 ledger heuristic baseline"
```

## What v1 does

v1 is still pure stdlib heuristics, but it adds a per-turn `GameState`, a 120-turn arrival ledger, and 1-ply candidate scoring. It predicts rotating planet and comet positions, estimates existing fleet arrivals, projects future garrisons and owner flips, then scores defense, expansion, comet, and attack candidates before launching.

## What v2 should add

v2 should add replay-driven tuning, stronger opponent pressure modeling, better multi-wave attack timing, and a compact opening book derived from local self-play.
