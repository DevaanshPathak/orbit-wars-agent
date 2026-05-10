# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A Kaggle competition agent for **Orbit Wars** (real-time strategy on a 100x100 board with rotating planets, comets, and a destructive central sun). The Kaggle submission must be a single `main.py` with a top-level `agent(obs, config=None)` function and **no external imports beyond `kaggle_environments`** — training dependencies (PyTorch, huggingface_hub, etc.) must never leak into the submitted file.

The repo is organized around iterative versions (v0–v9 in `CHANGELOG.md`). v0–v4 were pure heuristic; v5+ layer a learned candidate ranker on top of the heuristic. v9 is the active version (scaled supervised counterfactual ensemble); v8 GRPO underperformed v7 publicly, so the RL path is shelved.

## Common commands

```bash
# Sanity check (runs main.py vs random for one game)
python main.py

# Local benchmarks
python test_local.py --games 50 --baselines random nearest starter greedy rusher
python test_local.py --games 50 --compare-git-ref <sha>     # compares main.py vs main.py at a past commit
python test_local.py --games 10 --four-player

# Generate v7 candidate dataset (CPU-bound, scale via --workers)
python generate_training_data.py --games 2500 --both-sides --workers 16 --max-candidates-per-turn 64
python generate_training_data.py --games 1 --max-rows 200 --one-side --no-upload  # smoke test

# Train (notebooks are the preferred path; CLIs below are equivalent)
python notebooks/v6/train_v6_ranker.py --csv data/<run_start_timestamp>/candidates_v6.csv --upload
python notebooks/v7/train_v7_ranker.py --csv data/<run_start_timestamp>/candidates_v7.csv --upload
python notebooks/v9/train_v9_ranker.py --csv data/<run_start_timestamp>/candidates_v7.csv --upload

# Build a single-file Kaggle submission (embeds JSON weights into a copy of main.py)
python build_submission.py --weights notebooks/v9/exports/model_weights_v9.json --output models/v9_kaggle/main.py

# Submit
kaggle competitions submit orbit-wars -f models/v9_kaggle/main.py -m "v9 scaled ensemble"
```

There is no test suite, lint config, or CI. `test_local.py` is the validation harness — every code change should be checked against it (and ideally `--compare-git-ref` against the previous version) before any submission.

## Architecture

### `main.py` — single-file Kaggle agent (~2660 lines)

Per-turn flow inside `agent()`:

1. `GameState(obs)` builds an `_OBS_CACHE` of `Planet`/`Fleet` objects, comet IDs, and a precomputed `planet_path_cache` (positions over the next ~120 turns), then constructs an `ArrivalLedger` projecting all in-flight fleets onto each planet's future ownership/garrison timeline.
2. `_build_policy(state)` derives reaction times, indirect strategic value, per-source reserves, and the attack budget (ships freely available on each owned planet after defense reserves).
3. `_choose_moves(state, deadline)` runs a fixed pipeline against a hard time budget (`SPECULATIVE_TIME_MARGIN`):
   `_opening_planner_moves` → defense → evacuation → tactical (snipes/recaptures/crash exploits) → `_deep_planner_select` (beam search over capture bundles) → fallback greedy capture loop → optional staging.
4. Each candidate flows through `_validated_intercept` (which solves the moving-target intercept and re-checks the swept path against rotating planets and the sun) and `_apply_candidate` (which mutates `available` and `planned_commitments` so subsequent picks see the updated ledger).

**Critical invariant:** the heuristic-only path must remain Kaggle-safe. Two module-level constants control the learned model:

```python
USE_MODEL_SCORER = False
MODEL_WEIGHTS = None
```

In the source `main.py` these stay `False`/`None` so it runs as pure heuristic. `build_submission.py` does a literal string-replace on these two lines to embed the trained JSON weights — if you rename or restructure them, **you must update `build_submission.py` to match**, or submission builds will fail loudly.

`_model_score_candidate` handles three weight formats: `ensemble` (averages members), `mlp` (normalize → ReLU/tanh layers → sigmoid → scaled), and a logistic-regression fallback. The scorer **adds** a model bonus to the heuristic score blended by `MODEL_WEIGHTS["blend"]` (default 0.22) — it never replaces the heuristic, so a weak/missing model degrades gracefully instead of catastrophically.

### Candidate features → training data → ranker

`generate_training_data.py` reuses `main.py`'s candidate generators (it imports `main` directly), then for each turn it:

- Records up to `--max-candidates-per-turn` candidates with the full `FEATURE_FIELDS` schema (~80 features: phase, target, source, race, kind, rank context).
- Marks the actually-selected ones with `selected=1`.
- Adds **counterfactual positives** (high-ranked alternatives in losing games) and **failure tags** (`failure_overcommit`, `failure_missed_tactical`, `failure_missed_comet`, `failure_slow_expansion`).
- Computes turn-delta credit and future advantage deltas (5/15/30 turns ahead) for outcome-weighted labels.

Output: `data/<run_start_timestamp>/candidates_v7.csv` (one folder per run, all games multiplexed via `game_id`), uploaded to HF under the same path.

The training scripts (`notebooks/v6,v7,v9/train_v?_ranker.py`) consume this CSV. **Feature schema and metadata column lists must stay in lockstep** between `generate_training_data.py:FEATURE_FIELDS` and each trainer's `METADATA_COLS` — adding a new feature in the generator without updating the trainer's metadata exclusion list will make it leak metadata as a feature.

### Notebooks vs CLI trainers

Each `notebooks/v*/v*_training_policy.ipynb` is **self-contained** for Kaggle/Colab: it embeds the trainer code inline, prompts for `HF_TOKEN` via hidden input, downloads the newest HF `data/*/candidates_v7.csv` by default, streams epoch logs, and uploads artifacts to `devaanshpa/orbit-wars-agent/v?/`. The `train_v?_ranker.py` CLIs are the local-equivalent of those notebook cells; **the notebook embeds the `.py` as a string literal in cell-4 (`V9_RANKER_CODE`), so any edit to the `.py` must be mirrored in that string** — there is no runtime import path that would catch a drift. v8 SFT/GRPO and v9 also upload periodic per-member checkpoints during training (v8 every 30 epochs to `v8/sft/checkpoints` / `v8/grpo/checkpoints`; v9 every 40 epochs to `v9/checkpoints`).

The v8 SFT/GRPO trainers exist (`notebooks/v8/train_sft_policy.py`, `train_grpo_policy.py`) and run constrained policy learning over the same candidate generator — but per `roadmap.md`, v8 is on hold because GRPO smoke runs scored below v7 on the leaderboard.

## Hard repository rules (from `AGENTS.md`)

- **Never commit** trained models, checkpoints, exported weights (`.pt`, `.pth`, `.json` from `notebooks/*/exports/`, `model_weights_*.json`), replay datasets, or generated submission bundles. The `.gitignore` enforces most of this; `models/`, `data/`, `notebooks/**/exports/`, and `submission*.{zip,tar,tar.gz}` are all excluded.
- **Every new agent version updates `CHANGELOG.md`.** Versions that touch notebooks also update `notebooks/v*/notes.md`.
- Notebook authentication asks for `HF_TOKEN` at runtime via hidden input and must not print or persist it.
- HF artifact paths are versioned: `v5/`, `v6/`, `v7/`, `v8/sft/`, `v8/grpo/`, `v9/`. Datasets go to `data/<run_start_timestamp>/`. Downloaded model artifacts land in local `models/` (gitignored).
- For v7+ training, prefer `candidates_v7.csv` from the current `generate_training_data.py`. Don't train v7+ on older `candidates_v6.csv` unless explicitly running an ablation.
- Don't add `self` to `--opponents` as the default — it's slow self-play, only for explicit batches. Default mix is `random nearest starter greedy rusher`, both sides enabled.

## Environment notes

- Windows 11 + PowerShell is the primary dev environment. Use PowerShell syntax (`$env:VAR`, not `$VAR`; `;` not `&&`) when scripting.
- Python 3.11/3.12 recommended for training notebooks. Python 3.14 may break the pinned `torch>=2.3.0`; fall back to Kaggle/Colab if local install fails.
- Local data and exports can be large. The repo expects them to live under gitignored `data/`, `models/`, and `notebooks/**/exports/` only.
- `roadmap.md` is gitignored — it's a private planning doc with score targets and version gates. Read it for context but don't reference it in commits.
