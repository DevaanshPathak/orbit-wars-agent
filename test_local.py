from kaggle_environments import make

from main import agent


GAMES = 10


def run_game(seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([agent, "random"])
    final = env.steps[-1]
    return final[0].reward, final[1].reward


def main():
    wins = 0
    ties = 0
    losses = 0
    for seed in range(1, GAMES + 1):
        my_reward, opp_reward = run_game(seed)
        if my_reward > opp_reward:
            wins += 1
            result = "win"
        elif my_reward == opp_reward:
            ties += 1
            result = "tie"
        else:
            losses += 1
            result = "loss"
        print(f"seed={seed:02d} result={result} rewards=({my_reward}, {opp_reward})")

    win_rate = wins / GAMES
    print(f"wins={wins} ties={ties} losses={losses} win_rate={win_rate:.1%}")


if __name__ == "__main__":
    main()
