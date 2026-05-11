# v11 Notes

v11 reverts the aggressive heuristic changes from v10 (which scored 912.9, below v4's 934.9), keeps the wider beam search from v10, adds production-race escalation, and re-blends the v7 ensemble ranker on top.

## Files

- No new training notebook — v11 reuses the v7 model weights via `build_submission.py`.
- Model artifact: `models/v7_weights.json` (extracted from `models/v7_kaggle/main.py`, gitignored).

## What Changed from v10

### Reverted (hurt v10)
- `USE_STAGING = False` — staging sent ships to a forward planet without capturing, wasting a move turn
- Attack selection threshold restored to 18 (was 15 in v10)
- Comet selection threshold restored to 8, projected value multiplier back to 0.70, cap 22 turns, ETA penalty 0.35

### Kept from v10
- `PLANNER_BEAM = 4`, `PLANNER_MAX_PICKS = 4`, `PLANNER_TOP_CANDIDATES = 10` — wider beam search with no observed downside

### New in v11
- **Production-race escalation** in `_build_policy`: when `my_production / enemy_production < 0.85` after step 100, draws down planet reserves by up to 35% proportionally to the deficit, freeing ships for attacks instead of sitting idle.
- **Multi-source attack threshold** lowered from `production >= 4` to `production >= 3`: coordinated two-source attacks now trigger on more enemy planets.
- **v7 model re-blended**: `USE_MODEL_SCORER = True` with blend 0.22, built via `build_submission.py --weights models/v7_weights.json --output models/v11_kaggle/main.py`.

## Build Command

```bash
python build_submission.py --weights models/v7_weights.json --output models/v11_kaggle/main.py
```

## Submit Command

```bash
kaggle competitions submit orbit-wars -f models/v11_kaggle/main.py -m "v11 reverted heuristic + production-race + v7 model blend"
```

## Score Target

Beat v7's 1017.1 public score. The model alone should recover the ~82-point gain; production-race escalation and wider beam are the new upside.

## Lessons from v10

- **Staging is harmful at current tuning**: ships repositioning without capturing give the opponent a free productive turn.
- **Lower attack thresholds cause overcommits**: the 18.0 threshold was tuned; 15.0 led to attacks that failed to capture and wasted ships.
- **"More aggressive" ≠ "better"**: the v4 guarded approach is well-calibrated; improvements need to be smarter, not just more aggressive.
