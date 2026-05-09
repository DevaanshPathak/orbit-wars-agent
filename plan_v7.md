# v7 Plan: Replay-Guided Policy Improvement

## Goal

v7 should improve beyond v6 by learning from actual game outcomes, replay failures, and controlled local tournaments. The main objective is a higher Kaggle leaderboard score without replacing the safe heuristic action generator with a fragile full-policy model.

Target outcome:

- Beat v6 in local head-to-head tournaments by a statistically visible margin.
- Improve public Kaggle score after submission.
- Keep the Kaggle runtime single-file, stdlib-only, and comfortably under the 1 second turn limit.

## Why Not Full RL Yet

Full SFT + GRPO is still too expensive and risky for the next version:

- Orbit Wars actions are structured continuous moves, not simple discrete choices.
- Invalid actions and sun-crossing mistakes are easy for a direct policy model.
- Final reward is sparse and noisy across 500-turn games.
- The current heuristic candidate generator already handles geometry, legality, intercepts, sun safety, fleet sizing, and defense.

v7 should instead use model-based policy improvement: generate legal candidates with the existing engine, then learn better ranking and thresholds from outcomes and replay diagnostics.

## Phase 1: Replay and Failure Mining

Add tools to collect and analyze game outcomes:

- Save local tournament summaries to gitignored `data/<run_start_timestamp>/eval/`.
- Record per-game metadata: seed, opponent, side, result, reward margin, final ship totals, production totals, timeout/error status.
- Add optional replay JSON download/analysis for Kaggle episodes when submission IDs are available.
- Extract failure categories:
  - lost early expansion race
  - overcommitted homeland
  - missed comet
  - failed defense
  - bad attack into reinforcement
  - endgame score leak
  - sun/path loss or invalid launch

Output should be CSV/JSON artifacts only under gitignored `data/`, with summary tables printed to the console.

## Phase 2: Counterfactual Candidate Labels

Improve training labels beyond v6 outcome weighting:

- For each logged turn, store the candidate pool and selected candidate.
- For losing games, identify alternate high-ranking candidates that were available but not selected.
- Use shallow counterfactual rollouts where feasible:
  - replay the same seed with one candidate-selection knob changed
  - compare final margin against the original run
  - assign better labels to candidates associated with improved margins
- Add stronger labels for recurring tactical fixes:
  - defending planets that were later lost
  - earlier comet capture candidates
  - attacks that would have beaten projected reinforcement
  - endgame launches that preserve final score advantage

The v7 dataset should be `candidates_v7.csv` with all v6 features plus replay/failure features.

## Phase 3: Model and Planner Upgrades

Keep the candidate generator deterministic and legal, then improve ranking:

- Train a v7 MLP ranker on `candidates_v7.csv`.
- Compare against v6 with:
  - same feature set
  - added failure/replay features
  - separate early/mid/endgame heads if validation shows phase-specific behavior
- Export compact JSON weights under Hugging Face `v7/`.
- Update `main.py` only if the exported format changes; otherwise reuse the v6 MLP scorer.
- Tune runtime constants by local tournament:
  - model blend
  - planner beam/top candidates
  - defense reserve floors
  - comet urgency
  - endgame score mode thresholds

Do not commit trained weights, replay files, generated datasets, or submission bundles.

## Phase 4: Evaluation Harness

Add a repeatable local tournament command:

- Run v7 candidate builds against v6, v5, random, nearest, starter, greedy, and rusher.
- Support fixed seed ranges and both sides.
- Print win rate, average margin, median margin, timeout count, and result by opponent.
- Write machine-readable summaries under `data/<run_start_timestamp>/eval/`.

Acceptance criteria before Kaggle submission:

- No local runtime errors across at least 500 total games.
- No regression vs random/nearest/starter.
- Positive head-to-head margin vs v6 over mixed seeds.
- Model-enabled build compiles as a single `main.py`.

## Phase 5: RL Gate

Only start true RL-style optimization after the replay/counterfactual pipeline exists.

If v7 still plateaus, v8 can test a constrained RL loop:

- Policy acts only over generated legal candidates, never raw angles.
- Reward is final margin plus shaped intermediate production/planet-control deltas.
- Training uses self-play candidate selection, not direct move generation.
- Export remains a compact ranker, not a heavyweight model dependency.

This keeps RL inside the safe action scaffold instead of asking it to learn Orbit Wars legality from scratch.

## Immediate v7 Inputs Needed

Use these artifacts from v6:

- `data/<run_start_timestamp>/candidates_v6.csv`
- `notebooks/v6/exports/metrics_v6.json`
- v6 pairwise metrics: `validation_turn_top1_selected_rate`, `validation_pair_accuracy`, and `validation_selected_mean_rank_fraction`
- local tournament results comparing v6 model build vs v5/v4
- Kaggle score and episode IDs after v6 submission

These inputs decide whether v7 should prioritize defense, comets, opening expansion, or endgame scoring.
