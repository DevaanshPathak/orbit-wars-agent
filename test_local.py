import argparse
import math
import subprocess
import types

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from main import agent


def obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


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
        ships_needed = nearest.ships + 1
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
                key=lambda p: (p.ships / max(1, p.production), math.hypot(mine.x - p.x, mine.y - p.y)),
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


def load_agent_from_git_ref(git_ref):
    source = subprocess.check_output(
        ["git", "show", f"{git_ref}:main.py"],
        text=True,
        encoding="utf-8",
    )
    module = types.ModuleType(f"agent_{git_ref.replace('/', '_')}")
    exec(compile(source, f"{git_ref}:main.py", "exec"), module.__dict__)
    return module.agent


def final_scores(final_state):
    obs = obs_get(final_state[0], "observation", None)
    if obs is None:
        obs = final_state[0].observation
    num_players = len(final_state)
    scores = [0] * num_players
    for planet in obs_get(obs, "planets", []):
        owner = planet[1]
        if owner != -1 and owner < num_players:
            scores[owner] += planet[5]
    for fleet in obs_get(obs, "fleets", []):
        owner = fleet[1]
        if owner != -1 and owner < num_players:
            scores[owner] += fleet[6]
    return scores


def run_two_player(seed, opponent):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([agent, opponent])
    final = env.steps[-1]
    return final[0].reward, final[1].reward, final_scores(final)


def run_four_player(seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([agent, "random", "starter", nearest_sniper])
    final = env.steps[-1]
    return [state.reward for state in final], final_scores(final)


def summarize_two_player(games, label, opponent, seed_start):
    wins = 0
    ties = 0
    losses = 0
    score_margin = 0
    for i in range(games):
        seed = seed_start + i
        my_reward, opp_reward, scores = run_two_player(seed, opponent)
        score_margin += scores[0] - scores[1]
        if my_reward > opp_reward:
            wins += 1
            result = "win"
        elif my_reward == opp_reward:
            ties += 1
            result = "tie"
        else:
            losses += 1
            result = "loss"
        print(
            f"{label:8s} seed={seed:03d} result={result:4s} "
            f"rewards=({my_reward}, {opp_reward}) scores={scores[:2]}"
        )

    win_rate = wins / games
    avg_margin = score_margin / games
    print(
        f"{label:8s} wins={wins} ties={ties} losses={losses} "
        f"win_rate={win_rate:.1%} avg_margin={avg_margin:.1f}"
    )
    return wins, ties, losses


def summarize_four_player(games, seed_start):
    wins = 0
    for i in range(games):
        seed = seed_start + i
        rewards, scores = run_four_player(seed)
        if rewards[0] == max(rewards) and rewards[0] > 0:
            wins += 1
            result = "win"
        else:
            result = "loss"
        print(f"mixed4   seed={seed:03d} result={result:4s} rewards={rewards} scores={scores}")
    print(f"mixed4   wins={wins} losses={games - wins} win_rate={wins / games:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=sorted(BASELINES),
        default=["random", "nearest"],
    )
    parser.add_argument("--four-player", action="store_true")
    parser.add_argument("--compare-git-ref")
    args = parser.parse_args()

    for baseline in args.baselines:
        summarize_two_player(args.games, baseline, BASELINES[baseline], args.seed_start)
    if args.compare_git_ref:
        old_agent = load_agent_from_git_ref(args.compare_git_ref)
        summarize_two_player(args.games, "gitref", old_agent, args.seed_start)
    if args.four_player:
        summarize_four_player(max(1, args.games // 2), args.seed_start)


if __name__ == "__main__":
    main()
