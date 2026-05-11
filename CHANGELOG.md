# Changelog

## v13 - Staging + Wider Beam + Safe Production-Race + Opportunity Scoring + Comet Extension

- Re-enabled staging (`USE_STAGING = True`) with a safety guard: skips staging if the front planet's `enemy_reach` ETA ≤ 15 turns, preventing ships from being moved into contested planets. One staging move per turn consolidates idle rear-planet ships toward the nearest safe front.
- Widened deep planner beam from 4 → 5 (`PLANNER_BEAM = 5`) for broader multi-target coverage.
- Tightened production-race drawdown to only strip reserves from safe planets (`enemy_reach` ETA > 20), leaving front-line planet garrisons intact during the aggression phase.
- Added opportunity scoring in `_candidate_score`: +`production * 8.0` bonus when an enemy planet has `ships < production * 3.5`, prioritizing recently weakened targets before their garrison recovers.
- Extended comet ETA cap 18 → 22 turns when `my_production >= enemy_production` so farther comets are contested when we're leading.
- Added `notebooks/v13/train_v13_ranker.py` and `notebooks/v13/v13_training_policy.ipynb` to train a fresh 8-member ensemble on the 5k-game `data/` dataset; cell 3 auto-discovers the newest HF dataset. Artifacts upload to `devaanshpa/orbit-wars-agent/v13`.

## v12 - Tighter Defense + Opponent Modeling + Production-Race Aggression + Retrained Ensemble

- Tightened proactive defense in `_build_policy`: `enemy_ships * 0.18` → `enemy_ships * 0.30`, so nearby threats reserve more ships before committing to attacks.
- Added gradual pre-endgame reserve ramp in `reserve_for`: releases up to 55% of reserves linearly over the 80 turns before `ENDGAME_STEP` (steps 325–405) instead of dropping to zero abruptly.
- Added one-ply opponent response modeling in `_planner_projected_value`: for each source planet in a candidate, if the enemy can capture it before the fleet lands, penalize by `source.production * remaining_turns * 1.2`.
- Strengthened production-race aggression in `_build_policy`: triggers earlier (step 80, was 100), wider threshold (ratio < 0.92, was 0.85), steeper ramp (`* 3.0`, was `* 2.0`), higher max drawdown (60%, was 35%) — when losing the production race, slashes reserves and forces expansion rather than passive defense.
- Added `notebooks/v12/train_v12_ranker.py` and `notebooks/v12/v12_training_policy.ipynb` to train a fresh 8-member ensemble on the 2500-game `data/20260510_141652/candidates_v7.csv` dataset; artifacts upload to `devaanshpa/orbit-wars-agent/v12`.

## v11 - Reverted Heuristic + Production-Race + v7 Model

- Reverted staging to `False` and all threshold/comet changes from v10 after v10 scored 912.9 (below v4's 934.9).
- Kept v10's wider deep planner beam (4 beams, 4 max picks, 10 top candidates per step).
- Added production-race escalation to `_build_policy`: when `my_production / enemy_production < 0.85` after step 100, draws down planet reserves proportionally (up to 35%) to free ships for attacks.
- Lowered multi-source attack threshold from `production >= 4` to `production >= 3` so coordinated two-source attacks trigger on more enemy planets.
- Re-blended v7 ensemble ranker (blend 0.22) on top of the reverted heuristic core via `build_submission.py`.

## v10 - Aggressive Heuristic

- Enabled staging (`USE_STAGING = True`): rear planets now consolidate ships to the nearest safe front planet as a last move each turn, reducing idle ships.
- Widened deep planner beam from 3 → 4 beams, increased max picks from 3 → 4, and raised top-candidates per step from 8 → 10, giving the beam search broader multi-target coverage within the same time budget.
- Lowered attack selection threshold from 18 → 15, making the agent more willing to commit to enemy-planet attacks.
- Lowered comet selection threshold from 8 → 6, boosted comet projected value multiplier from 0.70 → 0.90 with extended cap (22 → 25 turns), and reduced comet ETA penalty (0.35 → 0.20) to prioritize contested comets more aggressively.
- Fixed `train_v9_ranker.py` to call `load_dotenv()` at the start of `main()` so `HF_TOKEN` is loaded from `.env` even when `--csv` is provided and `find_training_csv()` is bypassed.

## v9 - Scaled Counterfactual Ensemble Ranker

- Removed the experimental TPU SFT/GRPO v9 path after the v8 GRPO smoke submission underperformed the supervised v7 ranker.
- Added `notebooks/v9/train_v9_ranker.py`, a scaled supervised ensemble trainer based on the v7 counterfactual ranking method.
- Added `notebooks/v9/v9_training_policy.ipynb`, a self-contained streaming notebook runner that embeds the trainer code, asks for `HF_TOKEN`, downloads the newest Hugging Face `data/*/candidates_v7.csv` by default, and uploads artifacts to `devaanshpa/orbit-wars-agent/v9`.
- Set v9 defaults for the next 2500-game both-sides dataset: 8 ensemble members, 280 epochs, 4096 batch size, 1.05 pair-loss weight, 12 hard pairs per turn, and tuned blend export.
- Added per-member checkpoint export every 40 epochs in v9 training; checkpoints upload to `devaanshpa/orbit-wars-agent/v9/checkpoints/` when `--upload` is enabled (configurable via `V9_CHECKPOINT_EVERY` / `--checkpoint-every`).
- Removed the placeholder v10 notebook folder so the next version starts from a clean plan.

## v8 - Constrained SFT + GRPO Policy Workbench

- Added `notebooks/v8` with SFT and GRPO training entrypoints that keep Orbit Wars legality inside the existing candidate generator.
- Added a listwise SFT trainer that learns per-turn candidate selection from v7/v8 candidate CSV groups and exports Kaggle-compatible JSON MLP weights.
- Added a constrained GRPO-style policy improvement trainer that starts from the SFT artifact, samples legal candidates, applies group-relative advantages, and uploads artifacts under the requested Hugging Face paths.
- Added v8 notebook notes and streaming notebook launchers so SFT and GRPO runs ask for `HF_TOKEN`, show logs, save graphs, and avoid GitHub-tracked model outputs.
- Updated the GRPO path to download the SFT artifact from Hugging Face by default using `HF_TOKEN`, with local SFT JSON only as an explicit override.
- Updated the SFT path to download the newest Hugging Face `data/*/candidates_v7.csv` by default using `HF_TOKEN`, with local CSV only as an explicit override.
- Moved v8 Hugging Face uploads from the legacy `v7/sft` and `v7/grpo` experiment paths to `v8/sft` and `v8/grpo`.
- Switched v8 back to Kaggle 2*T4/CUDA defaults, with `V8_DEVICE=cuda` and no `torch_xla` requirement in the v8 notebooks.
- Set explicit v8 notebook training defaults for 1000-game both-sides datasets on Kaggle 2*T4, including SFT epochs/ensemble/batch settings and GRPO KL/anchor/batch settings.
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
