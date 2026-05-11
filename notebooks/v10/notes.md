# v10 Notes

v10 improves the heuristic core directly rather than adding more model training. The v9 5k-game ensemble regressed from v7 (959 vs 1017), and the gap to the leaderboard top (~636 points) is too large to close with a model layered on a weak heuristic. v10 targets the heuristic fundamentals.

## Files

- No training notebook for v10 — submit `main.py` directly.
- If a model is retrained after the heuristic baseline is established, add a `train_v10_ranker.py` and `v10_training_policy.ipynb` here.

## Changes from v9/v4 baseline

### Staging (re-enabled)
`USE_STAGING = True`. Staging sends ships from rear planets to the nearest safe front planet as the last move each turn. Was disabled in v4 after early risk concerns; re-enabled with the existing guards (safe-front check, distance-ratio filter, `STAGING_MIN_SHIPS = 12`, `STAGING_MAX_ETA = 42`) now that the full capture pipeline sits ahead of it.

### Deep planner widened
- `PLANNER_BEAM`: 3 → 4
- `PLANNER_MAX_PICKS`: 3 → 4
- `PLANNER_TOP_CANDIDATES`: 8 → 10

Broader beam search finds better multi-target combinations within the same `PLANNER_BUDGET = 0.055` s time cap.

### Attack threshold lowered
`_selection_threshold` for `"attack"` kind: 18 → 15. Less conservative about committing ships to enemy planets.

### Comet priority boosted
- Selection threshold: 8 → 6
- Projected value multiplier in `_planner_projected_value`: 0.70 → 0.90
- Remaining-turns cap: 22 → 25
- ETA penalty: 0.35 → 0.20

Comets are temporary, contested, and produce ships — they should jump the queue ahead of lower-value neutrals.

## Run Order

1. Submit `main.py` directly (pure heuristic, no model weights needed):
   ```bash
   kaggle competitions submit orbit-wars -f main.py -m "v10 aggressive heuristic"
   ```
2. Check score vs v7 (1017.1) baseline. If regression, bisect: disable staging first, then revert planner widening.
3. If heuristic is stronger, optionally retrain the v9 ranker on fresh v10-generated data:
   ```bash
   python generate_training_data.py --games 2500 --both-sides --workers 16 --max-candidates-per-turn 64
   python notebooks/v9/train_v9_ranker.py --csv data/<run>/candidates_v7.csv --upload
   ```
4. Blend model back in via `build_submission.py` and compare scores.

## Score Target

Beat v7's 1017.1 public score. If staging or lower thresholds cause regression, revert and tune conservatively.

## Result

v10 scored **912.9** — regression from v4 (934.9). Staging and lower attack thresholds were net negative. v11 reverts those changes.
