# Changelog

## v6 - Outcome-Weighted Candidate Ranker

- Added outcome-weighted `candidates_v6.csv` generation so selected moves from winning games train as stronger positives and selected moves from losses are downweighted.
- Expanded default data generation to both player sides, the full local baseline mix, and more candidate rows per turn.
- Added `notebooks/v6` with a PyTorch MLP ranker, grouped validation split, training logs, graph export, and Hugging Face upload under `devaanshpa/orbit-wars-agent/v6`.
- Extended the agent model hook to score compact JSON-exported MLP artifacts while preserving the v5 logistic fallback.
- Added a local submission builder that embeds downloaded/exported model JSON into a gitignored `models/` submission file.
- Added pairwise within-turn ranker training and extra runtime features so larger v6 datasets improve candidate ordering rather than only row accuracy.

## v5 - Model-Guided Deep Planner

- Planned offline candidate scoring with the v5 training notebook under `notebooks/v5`.
- Integrated training evaluation and graph export into the v5 training notebook.
- Added `generate_training_data.py` to write timestamped local datasets under gitignored `data/` and upload them to Hugging Face under `data/<timestamp>/`.
- Added a runtime-safe v5 planner path that keeps `main.py` self-contained and falls back to v4 heuristics when no local model weights are available.
- Added Hugging Face artifact storage workflow for training outputs under `devaanshpa/orbit-wars-agent/v5`.
- Added repository rules that prohibit trained models, generated weights, replay datasets, and submission artifacts from being pushed to GitHub.

## v4 - Guarded Heuristic Agent

- Added guarded speculative logic with strict time checks.
- Disabled staging by default after score feedback showed risk from over-positioning.
- Tightened synchronized tactical sends so launch ship counts match intercept timing.
- Preserved crash cleanup, recapture, snipe, and score-aware endgame behavior behind feature flags.

## v3 - Planner Heuristic Agent

- Added a stronger opening planner for early expansion.
- Added tactical candidate passes for snipes, recaptures, and multi-player crash cleanup.
- Improved candidate selection around timing, claimed targets, and coordinated force sizing.

## v2 - Collision Race Heuristic Agent

- Added swept collision validation for launched fleets.
- Added race pressure checks against enemy reach.
- Added safer expansion, comet handling, evacuation, and endgame scoring behavior.

## v1 - Ledger Heuristic Baseline

- Added arrival ledger projection for planets and fleets.
- Improved defense, reinforcement, expansion, and attack sizing around future arrivals.
- Added coordinated attacks and more conservative homeland reserves.

## v0 - Heuristic Baseline

- Added orbit-aware planet prediction, comet prediction, sun avoidance, intercept targeting, and speed-curve-aware launch sizing.
- Added local benchmark harness and a single-file Kaggle-compatible `main.py`.
