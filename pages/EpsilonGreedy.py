from src.run_bandit import run_bandit_app
from src.bandits import EpsilonGreedy


def main():
    run_bandit_app(
        title="Multi-Armed Bandit: Epsilon-Greedy",
        bandit_cls=EpsilonGreedy,
        key="eps",
        log_filename="eps_logs.csv",
        params={
            "n_arms": (2, 10, 5),
            "batch": (1, 1000, 100),
            "auto_steps": (1, 1000, 100),
            "auto_delay": (0.1, 2.0, 0.3),
            "epsilon": (0.0, 1.0, 0.1),
        },
    )


if __name__ == "__main__":
    main()
