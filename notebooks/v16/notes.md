# v16 Notes

v16 is a **pure heuristic submission** — no model trained. It adds five targeted heuristic improvements on top of the v13/v15 heuristic baseline and fixes the training label logic for the next model version (v17).

## Files

- `notes.md` — this file. No trainer or notebook needed; the submission is built directly from `main.py`.

## What Changed (v15 → v16)

### B1 — Attack-timing bonus (`_candidate_score` + `_build_policy`)

New function `_enemy_departure_pressure(state)` scans every enemy fleet. If a fleet is moving *away* from an enemy planet (angle check), that planet's current garrison is lower than the ledger shows. `_candidate_score` adds `min(departed_ships * 0.35, production * 14.0)` bonus for `kind == "attack"` targets — we attack while the garrison is reduced.

### B2 — ETA filter in multi-source bundling (`_build_multi_attack_candidate`)

When building a coordinated multi-source attack, sources whose `src_eta > target_turn + 3` are now dropped. Previously stragglers with long travel times were included in bundles that would arrive long before them, wasting the "simultaneous arrival" advantage.

### B3 — Neutral-denial bonus (`_candidate_score`)

When `my_production > enemy_production * 1.10` after step 60, neutral planets near the enemy get a denial bonus: `+production * 6.0` base, ramping up to `+production * (28 - enemy_eta) * 0.22` per remaining turn until enemy arrival. Prevents the enemy from snowballing production off uncontested neutrals while we're leading.

### B4 — Late-endgame reserve strip (`_build_policy`)

After step 380, when we hold a 35%+ ship advantage and `enemy_total > 0`, reserves are stripped from planets with `enemy_reach` ETA > 12 turns. Ships that can't be threatened in the remaining ~20 turns convert directly to attack pressure.

### B5 — Staging enemy-fleet guard (`_generate_staging_candidates`)

Staging is now aborted if any enemy fleet will arrive at the destination front planet within 12 turns. Previously, staging could move ships into a planet that was about to be hit, handing the enemy free ships.

## Label Fix (A) — Not submitted as part of v16 heuristic

`generate_training_data.py` label formula split into win/tie/loss branches:

| Branch | Label formula |
|--------|---------------|
| Win (`result > 0`) | `0.53 + 0.24 * delta_signal + 0.08 * max(0, margin_unit)` |
| Loss (`result < 0`) | `0.30 + 0.14 * max(-1, delta_signal)` |
| Tie | `0.42 + 0.10 * delta_signal` |

Counterfactual thresholds tightened: `rank <= 4`, `score_gap >= -0.15`, `delta_15 < -5.0`. Expected `positive_rate` drops from ~70% to ~35%, giving the model a real discrimination signal.

## Heuristic Baseline

v13 heuristic (unchanged from v15):
- Staging re-enabled with contested-front guard (enemy ETA ≤ 15)
- Beam width: 5
- Per-planet production-race drawdown (safe planets only, enemy ETA > 20)
- Opportunity scoring for low-garrison enemy planets (+production × 8.0)
- Comet ETA cap extended to 22 when leading in production

## Submission

```bash
# Build and submit pure heuristic (no --weights flag):
python build_submission.py --output models/v16_kaggle/main.py
kaggle competitions submit orbit-wars -f models/v16_kaggle/main.py -m "v16 heuristic: attack timing + simultaneous arrival + neutral denial + endgame strip + staging guard"
```

## Score

| Submission | Score |
|-----------|-------|
| v13 heuristic | 998.5 |
| v15 (model, old labels) | 918.8 |
| **v16 heuristic (submitted 2026-05-17)** | pending |
