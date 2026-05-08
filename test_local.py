import argparse
import math

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from main import agent


def nearest_sniper(obs, config=None):
    del config
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]
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


BASELINES = {
    "random": "random",
    "starter": "starter",
    "nearest": nearest_sniper,
}


def run_two_player(seed, baseline):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([agent, BASELINES[baseline]])
    final = env.steps[-1]
    return final[0].reward, final[1].reward


def run_four_player(seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([agent, "random", "starter", nearest_sniper])
    final = env.steps[-1]
    return [state.reward for state in final]


def summarize_two_player(games, baseline, seed_start):
    wins = 0
    ties = 0
    losses = 0
    for i in range(games):
        seed = seed_start + i
        my_reward, opp_reward = run_two_player(seed, baseline)
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
            f"{baseline:8s} seed={seed:03d} result={result:4s} "
            f"rewards=({my_reward}, {opp_reward})"
        )

    win_rate = wins / games
    print(
        f"{baseline:8s} wins={wins} ties={ties} losses={losses} "
        f"win_rate={win_rate:.1%}"
    )
    return wins, ties, losses


def summarize_four_player(games, seed_start):
    wins = 0
    for i in range(games):
        seed = seed_start + i
        rewards = run_four_player(seed)
        if rewards[0] == max(rewards) and rewards[0] > 0:
            wins += 1
            result = "win"
        else:
            result = "loss"
        print(f"mixed4   seed={seed:03d} result={result:4s} rewards={rewards}")
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
    args = parser.parse_args()

    for baseline in args.baselines:
        summarize_two_player(args.games, baseline, args.seed_start)
    if args.four_player:
        summarize_four_player(max(1, args.games // 2), args.seed_start)


if __name__ == "__main__":
    main()
