import argparse
import atexit
import contextlib
import csv
import io
import json
import logging
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


HF_REPO_ID = "devaanshpa/orbit-wars-agent"
HF_REPO_TYPE = "model"
VERSION = "v7_counterfactual_teacher"
CSV_BASENAME = "candidates_v7.csv"
PROGRESS_BASENAME = "progress.txt"

make = None
Planet = None
main = None
_EXIT_STDERR_DEVNULL = None

FEATURE_FIELDS = [
    "step",
    "turns_left",
    "num_players",
    "my_planets",
    "enemy_planets",
    "neutral_planets",
    "planet_count_gap",
    "my_total",
    "enemy_total",
    "max_enemy_total",
    "total_ratio",
    "my_production",
    "enemy_production",
    "production_gap",
    "production_ratio",
    "phase_opening",
    "phase_midgame",
    "phase_lategame",
    "phase_endgame",
    "target_owner_neutral",
    "target_owner_enemy",
    "target_owner_projected_mine",
    "target_ships",
    "target_projected_garrison",
    "target_production",
    "target_high_production",
    "target_prod_per_ship",
    "target_prod_per_eta",
    "target_value_net",
    "target_value_density",
    "target_static",
    "target_orbiting",
    "target_comet",
    "eta",
    "eta_fraction_remaining",
    "ships_sent",
    "parts_count",
    "ships_per_eta",
    "ship_cost_fraction",
    "source_commit_fraction",
    "source_distance_min",
    "source_distance_avg",
    "source_reserve_min",
    "source_reserve_sum",
    "source_budget_sum",
    "source_budget_after_send",
    "enemy_eta",
    "enemy_ships",
    "enemy_pressure",
    "my_eta",
    "my_reach_ships",
    "my_reach_advantage",
    "race_margin",
    "race_margin_per_eta",
    "capture_margin",
    "capture_margin_ratio",
    "indirect_value",
    "heuristic_score_scaled",
    "turn_candidate_count",
    "heuristic_rank",
    "heuristic_rank_fraction",
    "heuristic_score_gap_to_top",
    "heuristic_score_gap_to_selected",
    "same_kind_rank_fraction",
    "eta_rank_fraction",
    "turn_advantage",
    "future_advantage_delta_5",
    "future_advantage_delta_15",
    "future_advantage_delta_30",
    "future_production_delta_15",
    "future_planet_delta_15",
    "phase_kind_expand_opening",
    "phase_kind_attack_endgame",
    "phase_kind_comet_midgame",
    "phase_kind_defend_under_pressure",
    "kind_expand",
    "kind_attack",
    "kind_comet",
    "kind_snipe",
    "kind_recapture",
    "kind_crash",
    "kind_stage",
    "kind_defend",
    "kind_evacuate",
]

METADATA_FIELDS = [
    "label",
    "selected",
    "outcome_weight",
    "game_result",
    "reward_margin",
    "agent_reward",
    "opponent_reward",
    "selected_heuristic_rank",
    "counterfactual_positive",
    "counterfactual_reason",
    "failure_overcommit",
    "failure_missed_tactical",
    "failure_missed_comet",
    "failure_slow_expansion",
    "game_id",
    "candidate_id",
    "version",
]

CSV_FIELDS = METADATA_FIELDS + FEATURE_FIELDS


@contextlib.contextmanager
def quiet_native_output(show_output=False):
    if show_output:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


@contextlib.contextmanager
def quiet_imports(show_output=False):
    if show_output:
        yield
        return
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with quiet_native_output(False), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(previous_disable_level)


def load_runtime(show_env_imports=False):
    global make, Planet, main
    if make is not None and Planet is not None and main is not None:
        return
    if not show_env_imports:
        atexit.register(suppress_native_stderr_at_exit)
    with quiet_imports(show_env_imports):
        from kaggle_environments import make as kaggle_make
        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet as KagglePlanet
        import main as agent_main
    make = kaggle_make
    Planet = KagglePlanet
    main = agent_main


def suppress_native_stderr_at_exit():
    global _EXIT_STDERR_DEVNULL
    if _EXIT_STDERR_DEVNULL is not None:
        return
    _EXIT_STDERR_DEVNULL = open(os.devnull, "w", encoding="utf-8")
    os.dup2(_EXIT_STDERR_DEVNULL.fileno(), 2)


def obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def load_dotenv(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value


def planets_from(obs):
    return [Planet(*p) for p in obs_get(obs, "planets", [])]


def nearest_sniper(obs, config=None):
    del config
    moves = []
    player = obs_get(obs, "player", 0)
    planets = planets_from(obs)
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not targets:
        return moves
    for mine in my_planets:
        nearest = min(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))
        ships_needed = int(nearest.ships + 1)
        if mine.ships >= ships_needed:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])
    return moves


def greedy_expander(obs, config=None):
    del config
    moves = []
    player = obs_get(obs, "player", 0)
    planets = planets_from(obs)
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    for mine in sorted(my_planets, key=lambda p: p.ships, reverse=True):
        available = max(0, int(mine.ships - max(3, mine.production * 2)))
        if available <= 0 or not targets:
            continue
        target = max(
            targets,
            key=lambda p: (
                p.production * 18.0
                - p.ships
                - math.hypot(mine.x - p.x, mine.y - p.y) * 0.35
            ),
        )
        needed = int(target.ships + target.production * 3 + 2)
        if available >= needed:
            angle = math.atan2(target.y - mine.y, target.x - mine.x)
            moves.append([mine.id, angle, needed])
    return moves


def aggressive_rusher(obs, config=None):
    del config
    moves = []
    player = obs_get(obs, "player", 0)
    planets = planets_from(obs)
    my_planets = [p for p in planets if p.owner == player]
    enemies = [p for p in planets if p.owner not in (-1, player)]
    neutrals = [p for p in planets if p.owner == -1]
    for mine in sorted(my_planets, key=lambda p: p.ships, reverse=True):
        available = max(0, int(mine.ships - mine.production))
        if available <= 4:
            continue
        if enemies:
            target = min(enemies, key=lambda p: math.hypot(mine.x - p.x, mine.y - p.y))
            send = min(available, max(int(target.ships + 5), int(available * 0.72)))
        elif neutrals:
            target = min(
                neutrals,
                key=lambda p: (
                    p.ships / max(1, p.production),
                    math.hypot(mine.x - p.x, mine.y - p.y),
                ),
            )
            send = min(available, int(target.ships + 2))
        else:
            continue
        if send > 0:
            angle = math.atan2(target.y - mine.y, target.x - mine.x)
            moves.append([mine.id, angle, send])
    return moves


BASELINES = {
    "random": "random",
    "starter": "starter",
    "nearest": nearest_sniper,
    "greedy": greedy_expander,
    "rusher": aggressive_rusher,
    "self": None,
}


def resolve_opponent(name):
    if name not in BASELINES:
        raise ValueError(f"Unknown opponent {name!r}. Choose from {sorted(BASELINES)}")
    if name == "self":
        return main.agent
    return BASELINES[name]


def candidate_key(candidate):
    parts = ";".join(
        f"{int(source_id)}:{int(ships)}" for source_id, _, ships in candidate.parts
    )
    return (
        f"{candidate.kind}|{int(candidate.target_id)}|"
        f"{round(float(candidate.eta), 1)}|{int(candidate.ships)}|{parts}"
    )


def add_candidates(records, candidates, limit, score_fn):
    ranked = sorted(candidates, key=score_fn, reverse=True)
    for candidate in ranked[:limit]:
        key = candidate_key(candidate)
        records.setdefault(key, candidate)
    return ranked


def seed_candidate_pool(records, state, policy, max_candidates_per_turn):
    available = dict(policy["attack_budget"])
    planned = {}
    protected_targets = set()
    capture_limit = max_candidates_per_turn * 3
    tactical_limit = max_candidates_per_turn

    defense = main._generate_defense_candidates(state, available)
    add_candidates(records, defense, max_candidates_per_turn, lambda c: c.score)
    for candidate in defense:
        protected_targets.add(candidate.target_id)

    add_candidates(
        records,
        main._generate_evacuation_candidates(state, available, protected_targets),
        max_candidates_per_turn,
        lambda c: c.score,
    )

    tactical = (
        main._generate_snipe_candidates(state, available, planned, policy)
        + main._generate_recapture_candidates(
            state, available, planned, policy, protected_targets
        )
        + main._generate_crash_exploit_candidates(state, available, planned, policy)
    )
    add_candidates(records, tactical, tactical_limit, lambda c: c.score)

    capture = main._generate_capture_candidates(
        state,
        available,
        claimed_targets=set(),
        planned_commitments=planned,
        policy=policy,
    )
    add_candidates(
        records,
        capture,
        capture_limit,
        lambda c: main._score_candidate_v5(state, c, policy)
        + main._planner_projected_value(state, c, policy),
    )


def mark_if_applied(candidate, available, moves, planned, player, selected):
    if main._apply_candidate(candidate, available, moves, planned, player):
        selected.add(candidate_key(candidate))
        return True
    return False


def candidate_rows_from_obs(obs, game_id, max_candidates_per_turn, use_deep_planner):
    state = main.GameState(obs)
    if not state.my_planets:
        return []

    policy = main._build_policy(state)
    available = dict(policy["attack_budget"])
    moves = []
    claimed_targets = set()
    planned_commitments = {}
    protected_targets = set()
    records = {}
    selected = set()
    seed_candidate_pool(records, state, policy, max_candidates_per_turn)

    defense = add_candidates(
        records,
        main._generate_defense_candidates(state, available),
        max_candidates_per_turn,
        lambda c: c.score,
    )
    for candidate in defense:
        if mark_if_applied(
            candidate, available, moves, planned_commitments, state.player, selected
        ):
            claimed_targets.add(candidate.target_id)
            protected_targets.add(candidate.target_id)
        if len(moves) >= main.MAX_MOVES:
            break

    evacuation = add_candidates(
        records,
        main._generate_evacuation_candidates(state, available, protected_targets),
        max_candidates_per_turn,
        lambda c: c.score,
    )
    for candidate in evacuation:
        if mark_if_applied(
            candidate, available, moves, planned_commitments, state.player, selected
        ):
            claimed_targets.add(candidate.target_id)
        if len(moves) >= main.MAX_MOVES:
            break

    tactical = (
        main._generate_snipe_candidates(state, available, planned_commitments, policy)
        + main._generate_recapture_candidates(
            state, available, planned_commitments, policy, protected_targets
        )
        + main._generate_crash_exploit_candidates(
            state, available, planned_commitments, policy
        )
    )
    tactical = add_candidates(
        records,
        tactical,
        max_candidates_per_turn,
        lambda c: c.score,
    )
    for candidate in tactical:
        if candidate.target_id in claimed_targets and candidate.kind != "recapture":
            continue
        if candidate.score < main._selection_threshold(state, candidate):
            continue
        if mark_if_applied(
            candidate, available, moves, planned_commitments, state.player, selected
        ):
            claimed_targets.add(candidate.target_id)
        if len(moves) >= main.MAX_MOVES:
            break

    if use_deep_planner and len(moves) < main.MAX_MOVES:
        deadline = time.perf_counter() + 0.16
        planner_picks = main._deep_planner_select(
            state, available, claimed_targets, planned_commitments, policy, deadline
        )
        add_candidates(
            records,
            planner_picks,
            max_candidates_per_turn,
            lambda c: main._score_candidate_v5(state, c, policy)
            + main._planner_projected_value(state, c, policy),
        )
        for candidate in planner_picks:
            if candidate.target_id in claimed_targets:
                continue
            if main._score_candidate_v5(
                state, candidate, policy
            ) < main._selection_threshold(state, candidate):
                continue
            if mark_if_applied(
                candidate, available, moves, planned_commitments, state.player, selected
            ):
                claimed_targets.add(candidate.target_id)
            if len(moves) >= main.MAX_MOVES:
                break

    for _ in range(4):
        if len(moves) >= main.MAX_MOVES:
            break
        capture = main._generate_capture_candidates(
            state,
            available,
            claimed_targets,
            planned_commitments=planned_commitments,
            policy=policy,
        )
        if not capture:
            break
        capture = add_candidates(
            records,
            capture,
            max_candidates_per_turn,
            lambda c: main._score_candidate_v5(state, c, policy)
            + main._planner_projected_value(state, c, policy),
        )
        chosen = capture[0]
        if main._score_candidate_v5(state, chosen, policy) < main._selection_threshold(
            state, chosen
        ):
            break
        if mark_if_applied(
            chosen, available, moves, planned_commitments, state.player, selected
        ):
            claimed_targets.add(chosen.target_id)
        else:
            claimed_targets.add(chosen.target_id)

    if not selected:
        return []

    rows = []
    for key, candidate in records.items():
        features = main._candidate_features(state, candidate, policy)
        was_selected = 1.0 if key in selected else 0.0
        row = {
            "label": was_selected,
            "selected": was_selected,
            "outcome_weight": 1.0,
            "game_result": 0.0,
            "reward_margin": 0.0,
            "agent_reward": 0.0,
            "opponent_reward": 0.0,
            "game_id": game_id,
            "candidate_id": key,
            "version": VERSION,
        }
        for field in FEATURE_FIELDS:
            row[field] = float(features.get(field, 0.0))
        rows.append(row)
    return rows


def final_rewards_from_env(env, side):
    final_step = env.steps[-1] if env.steps else []
    rewards = []
    for state in final_step:
        reward = getattr(state, "reward", None)
        rewards.append(0.0 if reward is None else float(reward))
    agent_reward = rewards[side] if side < len(rewards) else 0.0
    opponent_rewards = [reward for index, reward in enumerate(rewards) if index != side]
    opponent_reward = max(opponent_rewards) if opponent_rewards else 0.0
    margin = agent_reward - opponent_reward
    if margin > 1e-9:
        result = 1.0
    elif margin < -1e-9:
        result = -1.0
    else:
        result = 0.0
    return agent_reward, opponent_reward, margin, result


def row_float(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def is_kind(row, *names):
    return any(row_float(row, f"kind_{name}", 0.0) >= 0.5 for name in names)


def evaluate_obs_advantage(obs):
    planets = planets_from(obs)
    fleets = [main.Fleet(*f) for f in obs_get(obs, "fleets", [])]
    player = obs_get(obs, "player", 0)
    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner not in (-1, player)]
    my_planet_ships = sum(float(p.ships) for p in my_planets)
    enemy_planet_ships = sum(float(p.ships) for p in enemy_planets)
    my_fleet_ships = sum(float(f.ships) for f in fleets if f.owner == player)
    enemy_fleet_ships = sum(float(f.ships) for f in fleets if f.owner not in (-1, player))
    my_production = sum(float(p.production) for p in my_planets)
    enemy_production = sum(float(p.production) for p in enemy_planets)
    my_planet_count = len(my_planets)
    enemy_planet_count = len(enemy_planets)
    ship_gap = (my_planet_ships + 0.55 * my_fleet_ships) - (enemy_planet_ships + 0.55 * enemy_fleet_ships)
    production_gap = my_production - enemy_production
    planet_gap = my_planet_count - enemy_planet_count
    advantage = ship_gap + production_gap * 24.0 + planet_gap * 18.0
    return {
        "advantage": advantage,
        "production_gap": production_gap,
        "planet_gap": float(planet_gap),
    }


def _future_delta(step_values, step, key, horizon):
    if not step_values:
        return 0.0
    future_steps = [known_step for known_step in step_values if known_step >= step + horizon]
    if future_steps:
        future_step = min(future_steps)
    else:
        future_step = max(step_values)
    return float(step_values[future_step].get(key, 0.0)) - float(step_values.get(step, {}).get(key, 0.0))


def reason_for_counterfactual(row, selected_row, result, delta_15):
    if result >= 0.0 and delta_15 >= -6.0:
        return ""
    if is_kind(row, "defend", "recapture", "evacuate"):
        return "missed_tactical"
    if is_kind(row, "comet") and row_float(row, "eta", 999.0) <= 24.0:
        return "missed_comet"
    if is_kind(row, "expand") and row_float(row, "phase_opening") >= 0.5:
        if row_float(row, "target_prod_per_eta") >= row_float(selected_row, "target_prod_per_eta", -999.0):
            return "slow_expansion"
    if is_kind(row, "attack", "snipe", "crash") and row_float(row, "capture_margin") > 1.0:
        return "missed_attack"
    return "higher_ranked_alternative"


def relabel_rows_with_outcome(rows, agent_reward, opponent_reward, margin, result, step_values=None):
    step_values = step_values or {}
    margin_scale = max(120.0, abs(margin))
    margin_unit = max(-1.0, min(1.0, margin / margin_scale))
    by_turn = defaultdict(list)
    for index, row in enumerate(rows):
        step = int(row_float(row, "step", 0.0))
        by_turn[(row.get("game_id", ""), step)].append((index, row))

    failure_counts = defaultdict(int)
    for _, items in by_turn.items():
        ordered = sorted(
            items,
            key=lambda item: row_float(item[1], "heuristic_score_scaled", -999.0),
            reverse=True,
        )
        selected_items = [item for item in ordered if row_float(item[1], "selected", 0.0) >= 0.5]
        selected_row = selected_items[0][1] if selected_items else ordered[0][1]
        selected_score = row_float(selected_row, "heuristic_score_scaled", 0.0)
        selected_rank = next(
            (rank for rank, (_, row) in enumerate(ordered, 1) if row is selected_row),
            len(ordered),
        )
        top_score = row_float(ordered[0][1], "heuristic_score_scaled", 0.0)
        by_eta = sorted(items, key=lambda item: row_float(item[1], "eta", 999.0))
        eta_ranks = {id(row): (rank + 1) / max(1.0, float(len(by_eta))) for rank, (_, row) in enumerate(by_eta)}
        same_kind = defaultdict(list)
        for item in items:
            same_kind[
                next((name for name in ("expand", "attack", "comet", "snipe", "recapture", "crash", "stage", "defend", "evacuate") if is_kind(item[1], name)), "other")
            ].append(item)
        same_kind_ranks = {}
        for kind_items in same_kind.values():
            kind_items.sort(key=lambda item: row_float(item[1], "heuristic_score_scaled", -999.0), reverse=True)
            for rank, (_, row) in enumerate(kind_items):
                same_kind_ranks[id(row)] = (rank + 1) / max(1.0, float(len(kind_items)))

        for rank, (_, row) in enumerate(ordered, 1):
            step = int(row_float(row, "step", 0.0))
            turn_advantage = float(step_values.get(step, {}).get("advantage", 0.0))
            delta_5 = _future_delta(step_values, step, "advantage", 5)
            delta_15 = _future_delta(step_values, step, "advantage", 15)
            delta_30 = _future_delta(step_values, step, "advantage", 30)
            production_delta_15 = _future_delta(step_values, step, "production_gap", 15)
            planet_delta_15 = _future_delta(step_values, step, "planet_gap", 15)
            delta_signal = max(-1.0, min(1.0, delta_15 / 120.0))
            selected = row_float(row, "selected", 0.0) >= 0.5
            score = row_float(row, "heuristic_score_scaled", 0.0)
            tactical = is_kind(row, "defend", "recapture", "evacuate")
            comet = is_kind(row, "comet")
            opening_expand = is_kind(row, "expand") and row_float(row, "phase_opening") >= 0.5
            expensive_commit = row_float(row, "source_commit_fraction") > 0.78 or row_float(row, "ship_cost_fraction") > 0.28
            counterfactual = False
            reason = ""

            if selected:
                if result > 0.0:
                    label = 0.53 + 0.24 * delta_signal + 0.08 * max(0.0, margin_unit)
                elif result < 0.0:
                    label = 0.30 + 0.14 * max(-1.0, delta_signal)
                else:
                    label = 0.42 + 0.10 * delta_signal
                weight = 0.70 + abs(delta_signal) * 1.40 + min(0.75, abs(margin_unit) * 0.55)
                if result == 0.0:
                    weight *= 0.25
                if delta_15 < -14.0 and expensive_commit and not tactical:
                    label = min(label, 0.22)
                    weight += 0.70
                    failure_counts["overcommit"] += 1
                if tactical and delta_15 >= -8.0:
                    label = max(label, 0.68)
                    weight += 0.55
            else:
                # Most unselected candidates are unknown, not true negatives. Keep them soft
                # and low-weight unless turn context says they were plausible corrections.
                label = max(0.08, min(0.42, 0.24 + 0.12 * (score - selected_score)))
                weight = 0.18 + min(0.35, max(0.0, top_score - score) * 0.08)
                hard_alternative = rank <= 4 and score >= selected_score - 0.15
                if hard_alternative and (delta_15 < -5.0 or result < 0.0):
                    reason = reason_for_counterfactual(row, selected_row, result, delta_15)
                    counterfactual = True
                    label = 0.56 + min(0.20, max(0.0, score - selected_score) * 0.08)
                    weight = 1.00 + min(1.30, abs(delta_signal) * 1.30 + abs(margin_unit) * 0.55)
                    if tactical:
                        label = 0.78
                        weight += 0.85
                        failure_counts["missed_tactical"] += 1
                    elif comet:
                        label = 0.68
                        weight += 0.45
                        failure_counts["missed_comet"] += 1
                    elif opening_expand:
                        label = 0.64
                        weight += 0.35
                        failure_counts["slow_expansion"] += 1

            row["label"] = round(max(0.0, min(1.0, label)), 6)
            row["outcome_weight"] = round(max(0.05, weight), 6)
            row["game_result"] = result
            row["reward_margin"] = round(margin, 6)
            row["agent_reward"] = round(agent_reward, 6)
            row["opponent_reward"] = round(opponent_reward, 6)
            row["selected_heuristic_rank"] = selected_rank
            row["counterfactual_positive"] = 1.0 if counterfactual else 0.0
            row["counterfactual_reason"] = reason
            row["failure_overcommit"] = 1.0 if selected and expensive_commit and delta_15 < -14.0 and not tactical else 0.0
            row["failure_missed_tactical"] = 1.0 if reason == "missed_tactical" else 0.0
            row["failure_missed_comet"] = 1.0 if reason == "missed_comet" else 0.0
            row["failure_slow_expansion"] = 1.0 if reason == "slow_expansion" else 0.0
            row["turn_candidate_count"] = len(ordered)
            row["heuristic_rank"] = rank
            row["heuristic_rank_fraction"] = round(rank / max(1.0, float(len(ordered))), 6)
            row["heuristic_score_gap_to_top"] = round(top_score - score, 6)
            row["heuristic_score_gap_to_selected"] = round(score - selected_score, 6)
            row["same_kind_rank_fraction"] = round(float(same_kind_ranks.get(id(row), 1.0)), 6)
            row["eta_rank_fraction"] = round(float(eta_ranks.get(id(row), 1.0)), 6)
            row["turn_advantage"] = round(turn_advantage, 6)
            row["future_advantage_delta_5"] = round(delta_5, 6)
            row["future_advantage_delta_15"] = round(delta_15, 6)
            row["future_advantage_delta_30"] = round(delta_30, 6)
            row["future_production_delta_15"] = round(production_delta_15, 6)
            row["future_planet_delta_15"] = round(planet_delta_15, 6)
            row["phase_kind_expand_opening"] = 1.0 if opening_expand else 0.0
            row["phase_kind_attack_endgame"] = 1.0 if is_kind(row, "attack") and row_float(row, "phase_endgame") >= 0.5 else 0.0
            row["phase_kind_comet_midgame"] = 1.0 if comet and row_float(row, "phase_midgame") >= 0.5 else 0.0
            row["phase_kind_defend_under_pressure"] = 1.0 if tactical and row_float(row, "enemy_pressure") > 0.45 else 0.0
            row["version"] = VERSION
    return dict(failure_counts)


class CsvRecorder:
    def __init__(self, writer, args):
        self.writer = writer
        self.args = args
        self.rows_written = 0
        self.positive_rows = 0
        self.turns_logged = 0
        self.current_game_id = ""

    def log(self, obs):
        if self.args.max_rows and self.rows_written >= self.args.max_rows:
            return
        rows = candidate_rows_from_obs(
            obs,
            self.current_game_id,
            self.args.max_candidates_per_turn,
            not self.args.no_deep_planner,
        )
        if not rows:
            return
        self.turns_logged += 1
        for row in rows:
            if self.args.max_rows and self.rows_written >= self.args.max_rows:
                break
            self.writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
            self.rows_written += 1
            if float(row["label"]) >= 0.5:
                self.positive_rows += 1


def run_games(args, recorder):
    opponents = [resolve_opponent(name) for name in args.opponents]
    game_count = 0
    for index in range(args.games):
        if args.max_rows and recorder.rows_written >= args.max_rows:
            break
        seed = args.seed_start + index
        opponent_name = args.opponents[index % len(args.opponents)]
        opponent = opponents[index % len(opponents)]
        recorder.current_game_id = f"seed_{seed}_p0_vs_{opponent_name}"

        def logging_agent(obs, config=None):
            recorder.log(obs)
            return main.agent(obs, config)

        with quiet_native_output(args.show_env_imports):
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run([logging_agent, opponent])
        game_count += 1

        if args.both_sides and (not args.max_rows or recorder.rows_written < args.max_rows):
            recorder.current_game_id = f"seed_{seed}_p1_vs_{opponent_name}"

            def logging_agent_p1(obs, config=None):
                recorder.log(obs)
                return main.agent(obs, config)

            with quiet_native_output(args.show_env_imports):
                env = make("orbit_wars", configuration={"seed": seed}, debug=False)
                env.run([opponent, logging_agent_p1])
            game_count += 1

        if game_count % args.progress_every == 0:
            print(
                f"games={game_count} rows={recorder.rows_written} "
                f"positive={recorder.positive_rows} turns={recorder.turns_logged}"
            )
    return game_count


def generate_game_rows(task):
    (
        seed,
        opponent_name,
        side,
        max_candidates_per_turn,
        use_deep_planner,
        show_env_imports,
    ) = task
    load_runtime(show_env_imports)
    opponent = resolve_opponent(opponent_name)
    game_id = f"seed_{seed}_p{side}_vs_{opponent_name}"
    rows = []
    step_values = {}
    turns_logged = 0
    start = time.perf_counter()

    def logging_agent(obs, config=None):
        nonlocal turns_logged
        step = int(obs_get(obs, "step", 0) or 0)
        step_values[step] = evaluate_obs_advantage(obs)
        turn_rows = candidate_rows_from_obs(
            obs,
            game_id,
            max_candidates_per_turn,
            use_deep_planner,
        )
        if turn_rows:
            turns_logged += 1
            rows.extend(turn_rows)
        return main.agent(obs, config)

    with quiet_native_output(show_env_imports):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        if side == 0:
            env.run([logging_agent, opponent])
        else:
            env.run([opponent, logging_agent])

    agent_reward, opponent_reward, margin, result = final_rewards_from_env(env, side)
    failure_counts = relabel_rows_with_outcome(
        rows, agent_reward, opponent_reward, margin, result, step_values
    )
    positives = sum(1 for row in rows if float(row["label"]) >= 0.5)
    counterfactuals = sum(
        1 for row in rows if row_float(row, "counterfactual_positive", 0.0) >= 0.5
    )
    return {
        "game_id": game_id,
        "seed": seed,
        "side": side,
        "opponent": opponent_name,
        "agent_reward": agent_reward,
        "opponent_reward": opponent_reward,
        "reward_margin": margin,
        "game_result": result,
        "rows": rows,
        "row_count": len(rows),
        "positive_rows": positives,
        "counterfactual_rows": counterfactuals,
        "failure_counts": failure_counts,
        "turns_logged": turns_logged,
        "duration_seconds": time.perf_counter() - start,
    }


def iter_tasks(args):
    for index in range(args.games):
        seed = args.seed_start + index
        opponent_name = args.opponents[index % len(args.opponents)]
        yield (
            seed,
            opponent_name,
            0,
            args.max_candidates_per_turn,
            not args.no_deep_planner,
            args.show_env_imports,
        )
        if args.both_sides:
            yield (
                seed,
                opponent_name,
                1,
                args.max_candidates_per_turn,
                not args.no_deep_planner,
                args.show_env_imports,
            )


def log_progress(done, total, rows, positives, counterfactuals, turns, started_at, last_result=None):
    elapsed = max(1e-6, time.time() - started_at)
    games_per_min = done / elapsed * 60.0
    rows_per_sec = rows / elapsed
    eta = (total - done) / max(1e-6, done / elapsed) if done else 0.0
    tail = ""
    if last_result is not None:
        result_text = (
            "win"
            if last_result.get("game_result", 0.0) > 0.0
            else "loss"
            if last_result.get("game_result", 0.0) < 0.0
            else "tie"
        )
        tail = (
            f" last={last_result['game_id']} "
            f"result={result_text} margin={last_result.get('reward_margin', 0.0):.1f} "
            f"rows={last_result['row_count']} cf={last_result.get('counterfactual_rows', 0)} "
            f"turns={last_result['turns_logged']} "
            f"time={last_result['duration_seconds']:.1f}s"
        )
    print(
        f"games={done}/{total} rows={rows} positive={positives} "
        f"counterfactual={counterfactuals} turns={turns} "
        f"rate={games_per_min:.2f}g/min {rows_per_sec:.1f}rows/s eta={eta/60.0:.1f}m"
        f"{tail}",
        flush=True,
    )


def upload_to_hf(run_dir, run_start_timestamp, repo_id, repo_type):
    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for upload. Add it to .env or set it in the environment.")
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install huggingface_hub to upload: pip install huggingface_hub") from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    remote_path = f"data/{run_start_timestamp}"
    api.upload_folder(
        folder_path=str(run_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=remote_path,
        commit_message=f"Upload Orbit Wars v7 training data {run_start_timestamp}",
    )
    return remote_path


def load_progress(run_dir):
    path = Path(run_dir) / PROGRESS_BASENAME
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            completed.add(line)
    return completed


def save_progress(run_dir, game_id):
    with (Path(run_dir) / PROGRESS_BASENAME).open("a", encoding="utf-8") as f:
        f.write(game_id + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Orbit Wars v7 counterfactual candidate training data.")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=sorted(BASELINES),
        default=["random", "nearest", "starter", "greedy", "rusher"],
    )
    parser.add_argument("--both-sides", action="store_true", default=True)
    parser.add_argument(
        "--one-side",
        action="store_false",
        dest="both_sides",
        help="Only log player 0 games. By default v7 logs both sides.",
    )
    parser.add_argument("--max-candidates-per-turn", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--run-start-timestamp",
        help="Folder name for this whole run. Defaults to the UTC run start time.",
    )
    parser.add_argument(
        "--timestamp",
        help="Deprecated alias for --run-start-timestamp.",
    )
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
    parser.add_argument("--hf-repo-type", default=HF_REPO_TYPE)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--no-deep-planner", action="store_true")
    parser.add_argument("--show-env-imports", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run. Requires --run-start-timestamp pointing to an existing run directory. Appends to the existing CSV and skips already-completed games.",
    )
    return parser.parse_args()


def main_cli():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    load_runtime(args.show_env_imports)
    run_start_timestamp = (
        args.run_start_timestamp
        or args.timestamp
        or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    run_dir = Path(args.output_root) / run_start_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / CSV_BASENAME

    completed_game_ids = set()
    if args.resume:
        completed_game_ids = load_progress(run_dir)
        if completed_game_ids:
            print(f"Resuming: {len(completed_game_ids)} games already done, will skip them.", flush=True)
        elif not csv_path.exists():
            print("Warning: --resume specified but no progress file found; starting fresh.", flush=True)

    started_at = time.time()
    total_tasks = args.games * (2 if args.both_sides else 1)
    rows_written = 0
    positive_rows = 0
    counterfactual_rows = 0
    turns_logged = 0
    games_run = 0
    failure_totals = defaultdict(int)
    csv_mode = "a" if (args.resume and completed_game_ids) else "w"
    with csv_path.open(csv_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if csv_mode == "w":
            writer.writeheader()
        if args.workers == 1:
            tasks = list(iter_tasks(args))
            if completed_game_ids:
                tasks = [t for t in tasks if f"seed_{t[0]}_p{t[2]}_vs_{t[1]}" not in completed_game_ids]
            total_tasks = len(tasks)
            print(
                f"starting workers=1 tasks={len(tasks)} "
                f"max_rows={args.max_rows or 'unlimited'}",
                flush=True,
            )
            for task in tasks:
                if args.max_rows and rows_written >= args.max_rows:
                    break
                game_result = generate_game_rows(task)
                rows_to_write = game_result["rows"]
                if args.max_rows:
                    rows_to_write = rows_to_write[: max(0, args.max_rows - rows_written)]
                for row in rows_to_write:
                    writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
                f.flush()
                save_progress(run_dir, game_result["game_id"])
                games_run += 1
                rows_written += len(rows_to_write)
                positive_rows += sum(
                    1 for row in rows_to_write if float(row["label"]) >= 0.5
                )
                counterfactual_rows += sum(
                    1 for row in rows_to_write if row_float(row, "counterfactual_positive", 0.0) >= 0.5
                )
                for key, value in game_result.get("failure_counts", {}).items():
                    failure_totals[key] += int(value)
                turns_logged += game_result["turns_logged"]
                if games_run % args.progress_every == 0 or games_run == total_tasks:
                    log_progress(
                        games_run,
                        total_tasks,
                        rows_written,
                        positive_rows,
                        counterfactual_rows,
                        turns_logged,
                        started_at,
                        game_result,
                    )
        else:
            tasks = list(iter_tasks(args))
            if completed_game_ids:
                tasks = [t for t in tasks if f"seed_{t[0]}_p{t[2]}_vs_{t[1]}" not in completed_game_ids]
            total_tasks = len(tasks)
            print(
                f"starting workers={args.workers} tasks={len(tasks)} "
                f"max_rows={args.max_rows or 'unlimited'}",
                flush=True,
            )
            pool_context = mp.get_context("spawn")
            with pool_context.Pool(processes=args.workers) as pool:
                for result in pool.imap_unordered(generate_game_rows, tasks):
                    if args.max_rows and rows_written >= args.max_rows:
                        pool.terminate()
                        break
                    rows_to_write = result["rows"]
                    if args.max_rows:
                        rows_to_write = rows_to_write[: max(0, args.max_rows - rows_written)]
                    for row in rows_to_write:
                        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
                    f.flush()
                    save_progress(run_dir, result["game_id"])
                    games_run += 1
                    rows_written += len(rows_to_write)
                    positive_rows += sum(
                        1 for row in rows_to_write if float(row["label"]) >= 0.5
                    )
                    counterfactual_rows += sum(
                        1 for row in rows_to_write if row_float(row, "counterfactual_positive", 0.0) >= 0.5
                    )
                    for key, value in result.get("failure_counts", {}).items():
                        failure_totals[key] += int(value)
                    turns_logged += result["turns_logged"]
                    if games_run % args.progress_every == 0 or games_run == total_tasks:
                        log_progress(
                            games_run,
                            total_tasks,
                            rows_written,
                            positive_rows,
                            counterfactual_rows,
                            turns_logged,
                            started_at,
                            result,
                        )
                    if args.max_rows and rows_written >= args.max_rows:
                        pool.terminate()
                        break

    manifest = {
        "version": VERSION,
        "run_start_timestamp": run_start_timestamp,
        "timestamp": run_start_timestamp,
        "games_requested": args.games,
        "games_run": games_run,
        "seed_start": args.seed_start,
        "opponents": args.opponents,
        "both_sides": args.both_sides,
        "workers": args.workers,
        "rows": rows_written,
        "positive_rows": positive_rows,
        "counterfactual_rows": counterfactual_rows,
        "failure_totals": dict(failure_totals),
        "turns_logged": turns_logged,
        "csv_path": str(csv_path),
        "csv_basename": CSV_BASENAME,
        "label_strategy": "turn_delta_counterfactual_pairwise_rank",
        "duration_seconds": round(time.time() - started_at, 3),
        "feature_fields": FEATURE_FIELDS,
        "metadata_fields": METADATA_FIELDS,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.no_upload:
        print(f"Saved local data to {run_dir}; upload skipped.")
        return not args.show_env_imports

    remote_path = upload_to_hf(
        run_dir, run_start_timestamp, args.hf_repo_id, args.hf_repo_type
    )
    print(f"Uploaded {run_dir} to https://huggingface.co/{args.hf_repo_id}/tree/main/{remote_path}")
    return not args.show_env_imports


if __name__ == "__main__":
    fast_exit = main_cli()
    if fast_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
