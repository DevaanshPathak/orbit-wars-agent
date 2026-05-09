# v6 Notes

v6 upgrades the v5 model path from imitation labels to outcome-weighted candidate ranking.

Training data should be generated with `generate_training_data.py`, which now writes `candidates_v6.csv` under one gitignored `data/<run_start_timestamp>/` folder and uploads that run to Hugging Face under `data/<run_start_timestamp>/`.

The v6 notebook trains a compact JSON-exportable MLP ranker using both outcome-weighted BCE and pairwise within-turn ranking loss, saves metrics and graphs under `notebooks/v6/exports/`, and uploads the export folder to `devaanshpa/orbit-wars-agent/v6`.

For larger datasets, prefer the pairwise trainer:

```bash
python notebooks/v6/train_v6_ranker.py --csv data/<run_start_timestamp>/candidates_v6.csv --upload
```

Generated data, exported weights, and submission bundles must stay out of git.
