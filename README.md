# Orbit Wars v0 Heuristic Agent

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

Run a small local benchmark against `random`:

```bash
python test_local.py
```

## Submit

```bash
kaggle competitions submit orbit-wars -f main.py -m "v0 heuristic baseline"
```

## What v0 does

v0 is a pure-heuristic baseline. It predicts rotating planet positions from `initial_planets`, leads targets with an iterative intercept solver, avoids firing through the sun, sizes long-distance fleets larger to exploit the log speed curve, defends threatened planets, captures efficient neutrals, opportunistically contests weak comets, and attacks high-production enemy planets when enough spare ships are available.

## What v1 should add

v1 should replace the target scoring with a stronger evaluation function, add a shallow one-ply outcome simulation for launches and arrivals, improve coordinated multi-source attacks, and add replay/log analysis tooling for tuning constants by map phase.
