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
from datetime import datetime, timezone
from pathlib import Path


HF_REPO_ID = "devaanshpa/orbit-wars-agent"
HF_REPO_TYPE = "model"
VERSION = "v5_teacher"

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
    "my_total",
    "enemy_total",
    "max_enemy_total",
    "my_production",
    "enemy_production",
    "production_gap",
    "target_owner_neutral",
    "target_owner_enemy",
    "target_owner_projected_mine",
    "target_ships",
    "target_projected_garrison",
    "target_production",
    "target_static",
    "target_orbiting",
    "target_comet",
    "eta",
    "ships_sent",
    "parts_count",
    "ships_per_eta",
    "ship_cost_fraction",
    "source_distance_min",
    "source_distance_avg",
    "source_reserve_min",
    "source_reserve_sum",
    "source_budget_sum",
    "enemy_eta",
    "enemy_ships",
    "my_eta",
    "my_reach_ships",
    "race_margin",
    "indirect_value",
    "heuristic_score_scaled",
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

CSV_FIELDS = ["label", "game_id", "candidate_id", "version"] + FEATURE_FIELDS


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
}


def resolve_opponent(name):
    if name not in BASELINES:
        raise ValueError(f"Unknown opponent {name!r}. Choose from {sorted(BASELINES)}")
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
        row = {
            "label": 1.0 if key in selected else 0.0,
            "game_id": game_id,
            "candidate_id": key,
            "version": VERSION,
        }
        for field in FEATURE_FIELDS:
            row[field] = float(features.get(field, 0.0))
        rows.append(row)
    return rows


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
    turns_logged = 0
    start = time.perf_counter()

    def logging_agent(obs, config=None):
        nonlocal turns_logged
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

    positives = sum(1 for row in rows if float(row["label"]) >= 0.5)
    return {
        "game_id": game_id,
        "seed": seed,
        "side": side,
        "opponent": opponent_name,
        "rows": rows,
        "row_count": len(rows),
        "positive_rows": positives,
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


def log_progress(done, total, rows, positives, turns, started_at, last_result=None):
    elapsed = max(1e-6, time.time() - started_at)
    games_per_min = done / elapsed * 60.0
    rows_per_sec = rows / elapsed
    eta = (total - done) / max(1e-6, done / elapsed) if done else 0.0
    tail = ""
    if last_result is not None:
        tail = (
            f" last={last_result['game_id']} "
            f"rows={last_result['row_count']} "
            f"turns={last_result['turns_logged']} "
            f"time={last_result['duration_seconds']:.1f}s"
        )
    print(
        f"games={done}/{total} rows={rows} positive={positives} turns={turns} "
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
        commit_message=f"Upload Orbit Wars v5 training data {run_start_timestamp}",
    )
    return remote_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Orbit Wars v5 candidate training data.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=sorted(BASELINES),
        default=["random", "nearest", "starter"],
    )
    parser.add_argument("--both-sides", action="store_true")
    parser.add_argument("--max-candidates-per-turn", type=int, default=18)
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
    csv_path = run_dir / "candidates_v5.csv"

    started_at = time.time()
    total_tasks = args.games * (2 if args.both_sides else 1)
    rows_written = 0
    positive_rows = 0
    turns_logged = 0
    games_run = 0
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        if args.workers == 1:
            recorder = CsvRecorder(writer, args)
            games_run = run_games(args, recorder)
            rows_written = recorder.rows_written
            positive_rows = recorder.positive_rows
            turns_logged = recorder.turns_logged
        else:
            tasks = list(iter_tasks(args))
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
                    games_run += 1
                    rows_written += len(rows_to_write)
                    positive_rows += sum(
                        1 for row in rows_to_write if float(row["label"]) >= 0.5
                    )
                    turns_logged += result["turns_logged"]
                    if games_run % args.progress_every == 0 or games_run == total_tasks:
                        log_progress(
                            games_run,
                            total_tasks,
                            rows_written,
                            positive_rows,
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
        "turns_logged": turns_logged,
        "csv_path": str(csv_path),
        "duration_seconds": round(time.time() - started_at, 3),
        "feature_fields": FEATURE_FIELDS,
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
