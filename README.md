# Orbit Wars Agent

This repository contains a single-file Orbit Wars Kaggle agent in `main.py`, plus training notebooks and data-generation tools for the heuristic and model-assisted versions.

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

## Generate training data

The generator runs local games, logs candidate features at decision time, adds turn-delta credit and counterfactual labels, writes one folder for the whole run under gitignored `data/<run_start_timestamp>/`, and uploads that run folder to Hugging Face under `devaanshpa/orbit-wars-agent/data/<run_start_timestamp>`.

```bash
python generate_training_data.py --games 2500 --both-sides --workers 16 --max-candidates-per-turn 64
```

Each game is identified inside `candidates_v7.csv` by `game_id`; games do not create separate timestamped folders. v7 defaults to both player sides and the full local baseline mix (`random nearest starter greedy rusher`). Add `self` to `--opponents` only for slower self-play runs.

For more visible progress and better CPU use on a multi-core machine:

```bash
python generate_training_data.py --games 2500 --both-sides --workers 16 --max-candidates-per-turn 64 --progress-every 5
```

For a local smoke test without upload:

```bash
python generate_training_data.py --games 1 --max-rows 200 --one-side --no-upload
```

The generator does not need a GPU. Orbit Wars game simulation and candidate extraction are CPU-bound Python work. Low total CPU usage usually means a single Python worker is saturating one core; use `--workers` to run independent games in parallel.

Kaggle may print optional OpenSpiel environment warnings in some installs. The generator suppresses that import noise by default; pass `--show-env-imports` if you need to debug Kaggle environment loading.

This writes `data/<run_start_timestamp>/candidates_v7.csv` and uploads the run folder to Hugging Face unless `--no-upload` is passed.

## Submit

After training v6 and downloading or keeping `model_weights_v6.json` locally, build a gitignored single-file agent:

```bash
python build_submission.py --weights notebooks/v6/exports/model_weights_v6.json --output models/v6_kaggle/main.py
```

```bash
kaggle competitions submit orbit-wars -f models/v6_kaggle/main.py -m "v6 outcome ranker"
```

## Hugging Face artifacts

Training notebooks write generated outputs under local `notebooks/**/exports/`, `models/`, and `data/` folders. These paths are gitignored. The v5 training notebook trains, evaluates, saves graphs, and uploads all exported artifacts to Hugging Face repo `devaanshpa/orbit-wars-agent` under the remote `v5/` folder after prompting for `HF_TOKEN`. The v6 notebook does the same under the remote `v6/` folder and trains a compact JSON-exportable MLP ranker.

Later local model downloads should go into the root `models/` folder, which is also gitignored.

Do not commit trained models, checkpoints, replay datasets, generated model exports, or submission bundles to GitHub.

## What v5 does

v5 keeps the v4 guarded heuristic core, then adds model-ready candidate features and a bounded deep planner over generated expansion, attack, and comet candidates. If no local model weights are available, the agent stays Kaggle-safe and falls back to heuristic scoring.

## What v6 does

v6 keeps the v5 candidate generator and planner, but trains on outcome-weighted labels instead of pure imitation labels. The v6 notebook trains a small MLP candidate ranker, exports it as JSON for single-file Kaggle submission builds, saves training graphs, and uploads artifacts to Hugging Face.

For a large CSV, the notebook runs the same pairwise trainer as this command:

```bash
python notebooks/v6/train_v6_ranker.py --csv data/<run_start_timestamp>/candidates_v6.csv --upload
```

The trainer optimizes both outcome-weighted classification and within-turn pairwise ranking, which is the metric that matters when the agent chooses among legal candidates.

## What v7 does

v7 adds a counterfactual dataset generator and an ensemble ranker trainer. Train it by running:

[notebooks/v7/v7_training_policy.ipynb](<notebooks/v7/v7_training_policy.ipynb>)

The notebook automatically uses the newest local `data/<run_start_timestamp>/candidates_v7.csv`, streams training logs live, and uploads artifacts to Hugging Face. If you prefer a shell command, it runs the same trainer as:

```bash
python notebooks/v7/train_v7_ranker.py --csv data/<run_start_timestamp>/candidates_v7.csv --upload
```

Then build a single-file submission:

```bash
python build_submission.py --weights notebooks/v7/exports/model_weights_v7.json --output models/v7_kaggle/main.py
```

The v7 model artifact is an ensemble of compact JSON MLP rankers; `main.py` can average ensemble members at runtime.

## What v8 does

v8 is the constrained SFT + GRPO workbench. It keeps the heuristic candidate generator as the legal action space, then trains Kaggle 2*T4/CUDA notebooks to rank those candidates:

- SFT notebook: [notebooks/v8/sft_training_policy.ipynb](<notebooks/v8/sft_training_policy.ipynb>)
- GRPO notebook: [notebooks/v8/grpo_training_policy.ipynb](<notebooks/v8/grpo_training_policy.ipynb>)

Both notebooks ask for `HF_TOKEN`, use CUDA on Kaggle GPU T4 x2, print every epoch, upload checkpoints every 30 epochs, and save final artifacts to Hugging Face under `v8/sft` and `v8/grpo`.

## What v9 does

v9 is a TPU v5e-8 training workbench. It keeps the same compact JSON model format, but trains SFT and GRPO-style reward-tuned ensemble members across TPU cores:

- SFT notebook: [notebooks/v9/sft_tpu_training_policy.ipynb](<notebooks/v9/sft_tpu_training_policy.ipynb>)
- GRPO notebook: [notebooks/v9/grpo_tpu_training_policy.ipynb](<notebooks/v9/grpo_tpu_training_policy.ipynb>)

Both notebooks are self-contained for Kaggle: they ask for `HF_TOKEN`, execute the embedded trainer in memory, use `torch_xla` on TPU, upload checkpoints every 30 epochs, and save final artifacts to Hugging Face under `v9/sft` and `v9/grpo`. They do not need a companion `.py` file in the Kaggle notebook session.

The current v9 defaults target a larger 2500-game both-sides dataset: 8 TPU members, 260 SFT epochs, 180 GRPO epochs, 8192-row and 8192-pair batches, and conservative KL anchoring for GRPO.
