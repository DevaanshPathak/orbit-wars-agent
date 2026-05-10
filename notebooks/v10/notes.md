# v10 Notes

v10 is the large-data push. It should use 3000+ games and start behaving less like heuristic imitation and more like a robust candidate-policy improvement system.

## Dataset Target

- 3000+ games with both sides enabled.
- Roughly 6000+ logged game perspectives.
- Mix baselines, prior versions, and targeted failure opponents.
- Include self-play only if the generator throughput is acceptable.
- Keep all datasets under `data/<run_start_timestamp>/`.
- Upload finished runs to Hugging Face under `data/<run_start_timestamp>/`.

## Goal

v10 should make the 1400-1600 public score range plausible.

Expected target:

- Strong positive margin against v8/v9 locally.
- Better comet timing, defense, and endgame score conversion.
- Less dependence on fixed heuristic weights.
- Public score target range: 1400-1600 if v9 already breaks past the heuristic ceiling.

## Data Mix

Prioritize diversity over just more rows:

- normal baseline games
- mirror-side games
- strong prior-version games
- loss-focused replay mining
- comet-heavy maps
- slow-expansion failure maps
- overcommit-defense failure maps
- endgame close-score maps

## SFT Direction

SFT should train from richer labels and harder candidate groups.

Suggested starting shape:

- SFT epochs: 220-320
- Batch groups: 256-384
- Ensemble size: 4-5
- Dropout: 0.14-0.22
- Patience: 32-40

Track:

- per-phase top1
- per-kind top1
- hard-negative pair accuracy
- calibration of model-vs-heuristic blend
- held-out game-family performance

## GRPO Direction

GRPO can be more ambitious at v10 scale.

Suggested starting shape:

- GRPO epochs: 160-240
- Batch groups: 192-320
- Samples per group: 12-16
- KL weight: 0.045-0.070
- Supervised anchor: 0.08-0.14
- Entropy weight: decay from 0.012 toward 0.004
- Patience: 28-40

At this scale, compare:

- SFT-only
- GRPO checkpoint
- distilled GRPO policy
- heuristic/model blend variants
- phase-specific blend variants

## Go / No-Go

v10 is worth submitting only if:

- It beats v9 locally by at least 55-60 percent head-to-head.
- It improves average final margin against v4-v9.
- It does not regress invalid actions, sun losses, or timing.
- Its improvement holds across held-out seeds and not just the training opponent mix.

If v10 does not clear these gates, the next step should be replay failure mining, not simply increasing epochs.
