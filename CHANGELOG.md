# Changelog

## v9 - TPU v5e-8 Candidate Policy Workbench

- Added `notebooks/v9/train_v9_tpu.py`, a TPU-first SFT/GRPO trainer that uses fixed-shape row and pair batches to avoid dynamic XLA recompilation.
- Added self-contained v9 Kaggle notebooks that write the embedded trainer at runtime, ask for `HF_TOKEN`, target TPU v5e-8 through `torch_xla`, and stream epoch logs.
- Added v9 SFT ensemble training across TPU cores with checkpoint uploads under `devaanshpa/orbit-wars-agent/v9/sft/checkpoints`.
- Added v9 GRPO-style reward tuning across TPU cores with checkpoint uploads under `devaanshpa/orbit-wars-agent/v9/grpo/checkpoints`.
- Added v9 artifact rules and notes for final JSON exports under `v9/sft` and `v9/grpo`.

## v8 - Constrained SFT + GRPO Policy Workbench

- Added `notebooks/v8` with SFT and GRPO training entrypoints that keep Orbit Wars legality inside the existing candidate generator.
- Added a listwise SFT trainer that learns per-turn candidate selection from v7/v8 candidate CSV groups and exports Kaggle-compatible JSON MLP weights.
- Added a constrained GRPO-style policy improvement trainer that starts from the SFT artifact, samples legal candidates, applies group-relative advantages, and uploads artifacts under the requested Hugging Face paths.
- Added v8 notebook notes and streaming notebook launchers so SFT and GRPO runs ask for `HF_TOKEN`, show logs, save graphs, and avoid GitHub-tracked model outputs.
- Updated the GRPO path to download the SFT artifact from Hugging Face by default using `HF_TOKEN`, with local SFT JSON only as an explicit override.
- Updated the SFT path to download the newest Hugging Face `data/*/candidates_v7.csv` by default using `HF_TOKEN`, with local CSV only as an explicit override.
- Moved v8 Hugging Face uploads from the legacy `v7/sft` and `v7/grpo` experiment paths to `v8/sft` and `v8/grpo`.
- Switched the v8 SFT and GRPO notebooks to TPU-first execution with `PJRT_DEVICE=TPU`, `V8_DEVICE=tpu`, `torch_xla` dependency checks, and XLA optimizer steps.
- Set explicit v8 notebook training defaults for 1000-game both-sides datasets on Kaggle TPU v5e-8, including SFT epochs/ensemble/batch settings and GRPO KL/anchor/batch settings.
- Made the self-contained v8 notebooks log every epoch and upload compact JSON checkpoints to Hugging Face every 30 epochs by default.

## v7 - Counterfactual Ensemble Ranker

- Updated `generate_training_data.py` to produce `candidates_v7.csv` with turn-delta credit, counterfactual positives, and failure metadata for overcommit, missed tactical moves, missed comets, and slow openings.
- Added `notebooks/v7` with a streaming notebook launcher and an ensemble MLP ranker trainer that uploads artifacts under `devaanshpa/orbit-wars-agent/v7`.
- Extended the runtime model scorer to average ensemble JSON artifacts while keeping trained weights out of GitHub.
- Tuned v7 defaults for 1000-game/both-sides datasets: larger candidate pool, stronger pairwise loss, larger batches, longer patience, and a 4-member ensemble.

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
