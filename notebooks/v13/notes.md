# v13 Notes

v13 is a **pure heuristic submission** — no model, no training. It tests whether the five heuristic improvements introduced in this version improve the score on their own before a learned ranker is layered on top in v14.

## Submission

Submitted `main.py` directly (97KB vs ~14MB for a model build):

```bash
kaggle competitions submit orbit-wars -f main.py -m "v13 pure heuristic: staging + wider beam + safe production-race + opportunity scoring + comet extension"
```

## Heuristic Changes (main.py)

### Staging re-enabled
`USE_STAGING = True`. Ships on interior planets were completely idle. The staging pass pushes available ships from rear planets toward the nearest safe front planet (one move per turn). Added guard: if the front planet's `enemy_reach` ETA ≤ 15 turns, staging is skipped so ships aren't funnelled into an actively contested planet.

### Wider beam search
`PLANNER_BEAM = 4` → `5`. Gives the beam search 25% more candidate bundles to keep per layer within the same time budget.

### Per-planet production-race drawdown
The production-race aggression block now skips planets where `enemy_reach` ETA ≤ 20 turns. Ships are only freed from safe interior planets; front-line planet garrisons are not stripped during the aggression phase.

### Opportunity scoring for low-garrison enemy planets
Added `+production * 8.0` bonus in `_candidate_score` when an enemy planet has `ships < production * 3.5`. Prioritizes striking planets that recently sent fleets out before their garrison recovers.

### Comet ETA extension when ahead
Comet ETA cap raised 18 → 22 turns when `my_production >= enemy_production`. When leading the production race, contesting farther comets is worthwhile.

## No Training

No model was trained for v13. The training notebook for the v13 heuristic has been promoted to v14 — see `notebooks/v14/`.
