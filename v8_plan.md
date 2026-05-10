# v8 Plan: Full RL via SFT + GRPO

## Goal

Build a reinforcement learning pipeline that can materially outperform the v7 ranker and push toward an aggressive Kaggle public score target of 1500. This is a stretch target, not a guaranteed score. The plan is to keep Orbit Wars legality and geometry inside the existing candidate generator, then train a stronger policy to select among legal candidate actions using supervised fine-tuning followed by GRPO-style policy improvement.

The final Kaggle submission must remain practical:

- Single `main.py` entrypoint.
- No trained weights committed to GitHub.
- Runtime under the 1 second turn budget.
- Fallback to the current heuristic or v7 blend if the RL policy is uncertain.

## Core Strategy

Do not train a raw continuous-action policy for v8. The action space is too easy to break: invalid launches, bad sun crossings, under-sized fleets, and impossible intercepts would waste most exploration.

Instead, v8 uses constrained RL:

- The existing heuristic engine generates legal candidate actions and action bundles.
- The model scores candidates within the current turn's candidate set.
- The sampled policy chooses one or more candidate bundles, with a no-op option when all candidates are poor.
- GRPO improves the candidate selector from full-game rewards and shaped tactical rewards.
- A compact distilled policy is exported for Kaggle inference.

This gives us RL pressure without throwing away the mechanics that already work: intercept targeting, fleet sizing, sun safety, comet timing, defense, and staged attacks.

## Local Layout

Add these local files for v8:

- `notebooks/v8/sft_training_policy.ipynb`
- `notebooks/v8/grpo_training_policy.ipynb`
- `notebooks/v8/notes.md`
- `notebooks/v8/train_sft_policy.py`
- `notebooks/v8/train_grpo_policy.py`
- Optional: `notebooks/v8/export_distilled_policy.py`

The notebooks must ask for `HF_TOKEN` in the first auth/setup cell using a hidden input, and must not print or save the token.

Per user request, trained artifacts from the v8 notebooks should upload to the Hugging Face repo `devaanshpa/orbit-wars-agent` under:

- `v8/sft/`
- `v8/grpo/`

## Data Plan

Use the improved v7 data generator as the base and extend it for RL.

Datasets stay gitignored under:

- `data/<run_start_timestamp>/`

Minimum useful dataset:

- 1000 games with both sides enabled.
- Opponent mix: random, nearest, starter, v6, v7, and optional self-play.
- Save every candidate set with:
  - candidate features
  - selected action id
  - action bundle details
  - turn context
  - current heuristic rank
  - local tactical label
  - future advantage delta
  - final reward margin
  - failure tags

Additional rollout data for GRPO:

- Store grouped rollouts by `(seed, side, opponent, map_signature)`.
- For each group, run multiple sampled policies from the same starting condition.
- Save per-rollout rewards, selected candidate ids, log probabilities, final margins, and shaped reward components.

## SFT Stage

SFT is the warm start. It should teach the model to imitate the best available teacher before RL exploration starts.

Inputs:

- `candidates_v7.csv` or a v8-compatible expanded candidate CSV.
- Replay-mined labels from wins and losses.
- Counterfactual labels where available.
- Per-turn candidate groups.

Model:

- Candidate encoder MLP over scalar candidate features.
- Turn context encoder over global game features.
- Candidate-set context using either:
  - DeepSets pooling over all candidates in the turn, or
  - a small attention block if export size stays manageable.
- Output:
  - policy logit for each candidate
  - optional value head for expected margin
  - optional no-op logit

Losses:

- Listwise cross entropy over candidates in the same turn.
- Pairwise ranking loss for strong positive-vs-hard-negative pairs.
- BCE or soft-label loss for tactical labels.
- Value regression to normalized future margin or shaped advantage.
- Entropy regularization to avoid overconfident early policies.

Validation metrics:

- Turn-level top-1 match against teacher.
- Mean selected rank fraction.
- Pairwise accuracy on hard pairs.
- Per-phase metrics: opening, expansion, midgame, endgame.
- Per-kind metrics: expand, attack, defend, comet, stage, snipe, crash, evacuate.

Artifacts uploaded to `v8/sft/`:

- model checkpoint
- compact JSON export
- feature schema
- metrics JSON
- training history
- loss and ranking graphs
- validation prediction samples

## GRPO Stage

GRPO should improve the SFT policy using full-game outcomes while keeping it close enough to the SFT policy to avoid collapse.

Rollout setup:

- Pick a fixed batch of seeds, opponents, and sides.
- For each environment setting, run a group of sampled policies from the same initial condition.
- Group size target:
  - 8 to 16 on TPU v5e-8 if environment rollout throughput supports it.
  - 4 to 8 only for a fallback 2*T4 debugging run.
- Sampling:
  - top-k or nucleus over legal candidates
  - temperature schedule from exploratory to conservative
  - hard legality checks after sampling
  - deterministic fallback if all sampled candidates are unsafe

Reward:

- Final normalized margin.
- Win/loss bonus.
- Production delta over time.
- Planet count delta over time.
- Comet ownership/capture bonus.
- Defense survival bonus for preventing captures.
- Penalty for sun-loss, invalid actions, overcommitment, and timeouts.
- Endgame ship-total reward.

GRPO objective:

- Normalize rewards within each rollout group.
- Compute advantage as each rollout reward minus the group mean.
- Optimize policy log probabilities for sampled actions using the group-relative advantage.
- Add KL penalty to the frozen SFT reference policy.
- Add entropy bonus early, decay later.
- Keep a value head only for diagnostics unless it improves stability.

Guardrails:

- Cap policy update size with KL threshold.
- Keep a rollback checkpoint whenever validation tournament score regresses.
- Blend with heuristic/v7 when model confidence is low.
- Reject any model that increases invalid actions, sun losses, or timeout rate.

Artifacts uploaded to `v8/grpo/`:

- GRPO checkpoint
- compact distilled JSON model
- rollout metrics
- reward component graphs
- KL and entropy graphs
- local tournament results
- export config used by `main.py`

## Hardware Plan

TPU v5e-8:

- Default runtime for the v8 notebooks.
- Best for larger SFT batches and high-throughput model updates.
- Rollout speed may still be CPU-bound because Kaggle environment simulation is Python-heavy.
- Use TPU mainly for training batches; keep environment workers on CPU.
- Notebooks should set `PJRT_DEVICE=TPU`, require `torch_xla`, and use XLA optimizer steps.

2*T4 GPU:

- Fallback runtime only if TPU/XLA setup fails.
- Use larger gradient accumulation instead of very large batches.

Recommendation:

- Start SFT and GRPO on TPU v5e-8.
- Fall back to 2*T4 only for debugging dependency or XLA issues.

## Distillation and Kaggle Export

The training model can be larger than the Kaggle model, but the submission model must be compact.

Export path:

1. Train SFT teacher.
2. Improve with GRPO.
3. Generate a large offline dataset of GRPO policy decisions.
4. Distill into a small inference model:
   - tabular MLP
   - optional ensemble of 3 to 5 compact models
   - JSON weights only
5. Fetch artifacts into local gitignored `models/`.
6. Build a Kaggle-safe `main.py` with embedded or bundled weights.
7. Run local tournaments before submission.

Inference blend:

- Tune global blend against heuristic.
- Tune phase-specific blend if validation supports it.
- Keep a confidence fallback:
  - low confidence: heuristic/v7 action
  - high confidence: v8 policy action

## Evaluation Gates

Do not submit v8 unless it passes these gates:

- Beats v7 head-to-head over at least 200 local games.
- Improves average final margin against v7.
- Beats random, nearest, starter, v6, and v7 in held-out seed tournaments.
- Keeps timeout rate at 0.
- Keeps invalid action rate at 0.
- Does not increase sun-loss rate materially.
- Shows improvement in at least two of:
  - opening expansion
  - comet capture
  - defense
  - attack timing
  - endgame scoring

Target local thresholds before Kaggle submission:

- 60 percent or better head-to-head win rate vs v7 on held-out seeds.
- 70 percent or better vs v6.
- 95 percent or better vs random/nearest/starter.
- Positive average margin in mixed-opponent tournament.

## Notebook Requirements

Both SFT and GRPO notebooks must:

- Ask for `HF_TOKEN` in the first setup cell.
- Install or verify dependencies.
- Locate the latest compatible CSV or allow `CANDIDATES_CSV` override.
- Print streaming logs during training.
- Save graphs at the end.
- Upload artifacts and graphs to Hugging Face.
- Avoid committing or saving models into GitHub-tracked paths.

SFT notebook should show:

- train and validation loss
- top-1 ranking metrics
- pairwise accuracy
- per-phase metrics
- top feature or attention diagnostics

GRPO notebook should show:

- rollout reward mean and standard deviation
- win rate by opponent
- KL to SFT policy
- entropy
- invalid/sun-loss/timeout counters
- reward component breakdown

## Risks

Main risks:

- RL overfits local opponents and drops Kaggle score.
- Sparse final rewards drown out real tactical signals.
- Rollouts are too slow for enough GRPO updates.
- Model learns to imitate heuristic errors if SFT labels are not strong enough.
- Larger models cannot be exported safely into Kaggle runtime.

Mitigations:

- Keep the legal candidate generator as the action space.
- Use shaped rewards and group-relative normalization.
- Keep SFT KL anchoring during GRPO.
- Distill to compact JSON for submission.
- Maintain v7 heuristic fallback.
- Evaluate on held-out seeds and opponent mixes before submitting.

## Implementation Order

1. Delete obsolete v7 planning doc and keep this v8 plan as the active RL plan.
2. Add v8 notebook skeletons with hidden HF token auth and upload helpers.
3. Extend data generation for grouped rollout logging.
4. Build SFT trainer and validate against v7 data.
5. Build GRPO rollout collector with small smoke runs.
6. Train GRPO from SFT checkpoint.
7. Distill GRPO policy to a compact export.
8. Wire the compact policy into `main.py`.
9. Run local tournaments against v6 and v7.
10. Submit only after evaluation gates are met.
