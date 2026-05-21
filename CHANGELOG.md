# Changelog

## v20 - Conservative Counterfactual RL Safety Gate

- Added `rl-counterfactual-v20` mode to `generate_training_data.py`; it writes `candidates_v20.csv`, uses longer default counterfactual rollouts, promotes positive rollout-delta alternatives, and downlabels selected candidates that crash or produce large negative rollout deltas.
- Added `notebooks/v20/` with self-contained SFT and GRPO notebooks plus optional local trainer scripts.
- v20 SFT defaults to the pinned uploaded `data/20260520_061012/candidates_v19.csv` dataset as an immediate fallback, while accepting `candidates_v20.csv` overrides from Hugging Face.
- v20 GRPO starts from the SFT baseline, rejects unsafe epochs via validation top-1 and KL gates, caps model blend, and carries SFT ensemble members into the final artifact to avoid the v19 reward-overfitting failure mode.
- Added v20 notebook notes documenting the recommended 2500-game v20 dataset command, HF upload paths, run order, and safety settings.

## v19 - Counterfactual RL

- Added `--mode rl-counterfactual` to `generate_training_data.py` to collect shallow causal rollouts (25-turn projections) and capture exact advantage deltas (`cf_margin_delta`, `cf_prod_delta`, `cf_planet_delta`) per candidate, replacing legacy static heuristics.
- Pinned v19 SFT/GRPO training to the counterfactual dataset at `devaanshpa/orbit-wars-agent/data/20260520_061012/candidates_v19.csv`.
- Added explicit experimental candidate generators (comet denial, split reserve expansion, delayed attacks) that fire during data generation to broaden the counterfactual action space.
- Added `notebooks/v19/train_sft_policy.py` which transitions target generation from static heuristic selected-labels to a counterfactual-weighted cross-entropy loss, aggressively prioritizing candidates that show positive causal rollout metrics.
- Added `notebooks/v19/train_grpo_policy.py` which updates the candidate reward components to directly use counterfactual deltas (`cf_margin_delta`, `cf_survival`, `cf_crash`) rather than just heuristic labels and static turn advantages.
- Upgraded Kaggle notebook shims (`sft_training_policy.ipynb` and `grpo_training_policy.ipynb`) for v19, keeping the 2*T4 setup with HF integration for `candidates_v19.csv` datasets.

## v18 - Reward-Resistant Constrained SFT + GRPO Workbench

- Added `notebooks/v18/` with self-contained SFT and GRPO notebooks plus optional local trainer scripts.
- SFT notebook downloads the pinned `data/20260517_074915/candidates_v7.csv`, trains a 4-member candidate-policy ensemble, logs every epoch, saves graphs, checkpoints every 30 epochs, and uploads artifacts to `devaanshpa/orbit-wars-agent/v18/sft/`.
- Updated v18 SFT to use `torch.nn.DataParallel` on multi-GPU CUDA runtimes by default (`V18_SFT_MULTI_GPU=1`) and flattened each batch of turn-groups into larger forwards so Kaggle 2*T4 has enough work to split.
- Updated v18 notebook auth cells to read `HF_TOKEN` from Kaggle Secrets or existing environment variables before falling back to a local prompt, avoiding Kaggle `getpass` stdin failures.
- GRPO notebook downloads the pinned candidate CSV and the SFT artifact from `v18/sft/model_weights_v18_sft.json`, then uploads final artifacts to `v18/grpo/`.
- Added bounded multi-component GRPO reward with reduced selected-action anchoring, future advantage deltas, production/planet deltas, final result/margin, explicit failure/crash/overcommit penalties, KL-to-SFT, supervised anchor, reward-gap diagnostics, and failure-exposure validation.
- Kept RL constrained to legal heuristic candidate move-sets rather than raw angle/ship generation, so the policy can improve strategic selection without relearning basic legality, sun avoidance, and intercept mechanics.

## v17 - v16 Heuristic + Pinned Offline Dataset Ensemble

- Uploaded the offline `data/20260517_074915` candidate dataset to Hugging Face under `devaanshpa/orbit-wars-agent/data/20260517_074915/`.
- Resolved the v17 merge conflict by keeping the self-contained v17 trainer/notebook path and pinning training to `data/20260517_074915/candidates_v7.csv`.
- `notebooks/v17/train_v17_ranker.py` now defaults to the pinned HF dataset when no `--csv` override is provided.
- `notebooks/v17/v17_training_policy.ipynb` is independent: it asks for `HF_TOKEN`, downloads the pinned dataset, trains, prints epoch logs, checkpoints, writes graphs, and uploads artifacts to `v17/` on HF.
- Model defaults remain the wider v17 supervised ensemble: 8 members, 256 -> 128 -> 64 MLP, 300 epochs, batch size 4096, pair loss 1.10, and checkpoint uploads every 40 epochs.

## v16.5 - Additional Heuristic Improvements on v16 Base

- Added two-ply chain bonus in `_planner_projected_value`: after estimating capture value for a target, scores the best adjacent capturable planet within 75 units and adds 20% of its projected value, teaching the planner to prefer positions that unlock follow-on captures.
- Added hot-recapture bonus in `_candidate_score`: enemy planets with garrison ≤ 2 ships (just captured) get `+production * 18 + 35` so recently lost territory is always retaken before the garrison grows.
- Added back-line attack bonus: when leading production by 20%+ after step 100, isolated enemy planets (nearest enemy support > 35 units) get `+production * 5 + distance * 0.08`, stretching the enemy's defense network instead of attacking reinforced front lines.
- Added adaptive comet density bonus: `+production * max(0, 20 - eta) * 0.30` rewards nearby high-production comets above distant low-production ones.
- Extended opening planner from step 50 → 60 and max-planet gate from 5 → 6; reduced orbiting penalty 10 → 5 and low-production penalty 10 → 6 so more planets are considered in early expansion.
- Triggered production-race aggression 10 turns earlier: `step > 70` (was `step > 80`).
- Improved multi-source attack ship allocation: sorted sources by ETA ascending and capped late-arriving sources to the remaining deficit once the first wave covers 70% of needed ships, avoiding wasted ships on already-captured planets.
- Pure heuristic submission (no model).

## v16 - Heuristic Fixes: Attack Timing + Simultaneous Arrival + Neutral Denial + Endgame Strip + Staging Guard

- Added `_enemy_departure_pressure` to `_build_policy`: detects enemy fleets moving away from a target planet and adds bonus score in `_candidate_score` (+`departed * 0.35`, capped at `production * 14`) so we attack while the garrison is reduced.
- Filtered simultaneous arrivals in `_build_multi_attack_candidate`: skips sources whose ETA exceeds `target_turn + 3`, preventing stragglers from being bundled into a group that arrives too early without them.
- Added neutral-denial bonus in `_candidate_score` when we lead production by 10%+ after step 60: +`production * 6.0` base, +`production * (28 - enemy_eta) * 0.22` when enemy is inbound within 28 turns.
- Added late-endgame reserve strip in `_build_policy`: after step 380, when we hold a 35%+ ship advantage, strips reserves from planets with `enemy_reach` ETA > 12 turns so idle ships convert to pressure.
- Added enemy-inbound staging guard in `_generate_staging_candidates`: aborts staging if any enemy fleet arrives at the destination front planet within 12 turns, preventing ships from being moved into an imminent fight.
- Fixed training labels in `generate_training_data.py`: selected candidates in losing games now receive soft-negative labels (~0.30) instead of ~0.48, breaking the circular replication pattern and reducing `positive_rate` from ~70% to ~35%.
- Widened `train_v15_ranker.py` hidden layers from `160→80→40` to `256→128→64` for better capacity on the 5k-game dataset.
- Pure heuristic submission path: build from `main.py` directly without model weights.

## v15 - v13 Heuristic + Retrained Ensemble on Pinned 5k-Game Dataset (20260516_032302)

- Trains a fresh 8-member ensemble ranker on top of the v13 heuristic using the pinned `data/20260516_032302` dataset (5k games × 2 sides = 10k game-plays, 608k rows, generated 2026-05-16).
- Added `notebooks/v15/train_v15_ranker.py` and `notebooks/v15/v15_training_policy.ipynb`; cell 3 fetches the **pinned** `data/20260516_032302/candidates_v7.csv` from HF directly (no auto-discovery). Artifacts upload to `devaanshpa/orbit-wars-agent/v15`.
- Added pause/resume support to `generate_training_data.py`: each completed game is appended to `progress.txt`; `--resume --run-start-timestamp <folder>` skips already-done games and appends to the existing CSV.

## v14 - v13 Heuristic + Retrained Ensemble on 5k-Game Dataset

- Trains a fresh 8-member ensemble ranker on top of the v13 heuristic using the 5k-game both-sides dataset.
- Added `notebooks/v14/train_v14_ranker.py` and `notebooks/v14/v14_training_policy.ipynb`; cell 3 auto-discovers the newest HF `data/*/candidates_v7.csv`. Artifacts upload to `devaanshpa/orbit-wars-agent/v14`.

## v13 - Staging + Wider Beam + Safe Production-Race + Opportunity Scoring + Comet Extension

- Re-enabled staging (`USE_STAGING = True`) with a safety guard: skips staging if the front planet's `enemy_reach` ETA ≤ 15 turns, preventing ships from being moved into contested planets. One staging move per turn consolidates idle rear-planet ships toward the nearest safe front.
- Widened deep planner beam from 4 → 5 (`PLANNER_BEAM = 5`) for broader multi-target coverage.
- Tightened production-race drawdown to only strip reserves from safe planets (`enemy_reach` ETA > 20), leaving front-line planet garrisons intact during the aggression phase.
- Added opportunity scoring in `_candidate_score`: +`production * 8.0` bonus when an enemy planet has `ships < production * 3.5`, prioritizing recently weakened targets before their garrison recovers.
- Extended comet ETA cap 18 → 22 turns when `my_production >= enemy_production` so farther comets are contested when we're leading.
- Pure heuristic submission — no model trained. Training notebook for this heuristic is in `notebooks/v14/` (promoted to v14 to keep version numbers aligned with submissions).

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
