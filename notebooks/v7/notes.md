# v7 Notes

v7 trains from `candidates_v7.csv`, which adds counterfactual positives for high-ranking alternatives in losing games.

The dataset generator is:

```bash
python generate_training_data.py --games 1000 --workers 16 --max-candidates-per-turn 48
```

The preferred model generator is the notebook:

```bash
notebooks/v7/v7_training_policy.ipynb
```

It automatically picks the newest local `candidates_v7.csv`, runs the trainer with unbuffered output, shows live epoch logs, and uploads artifacts.

The direct shell equivalent is:

```bash
python notebooks/v7/train_v7_ranker.py --csv data/<run_start_timestamp>/candidates_v7.csv --upload
```

The trainer exports an ensemble JSON artifact under `notebooks/v7/exports/model_weights_v7.json` and uploads artifacts to `devaanshpa/orbit-wars-agent/v7`.

Generated data, exported weights, and submission bundles must stay out of git.
