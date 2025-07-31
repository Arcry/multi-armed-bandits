from src.run_bandit import run_bandit_app
from src.bandits import UCB1


def main():
    run_bandit_app(
        title="Multi-Armed Bandit: UCB1",
        bandit_cls=UCB1,
        key="ucb",
        log_filename="ucb_logs.csv",
        params={
            "n_arms": (2, 10, 5),
            "batch": (1, 1000, 100),
            "auto_steps": (1, 1000, 100),
            "auto_delay": (0.1, 2.0, 0.3),
            # no epsilon/prior here—just the common sliders
        },
    )


if __name__ == "__main__":
    main()
