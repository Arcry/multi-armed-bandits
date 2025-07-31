import streamlit as st
from src.run_bandit import run_bandit_app
from src.bandits import Thompson


def main():
    run_bandit_app(
        title="Multi-Armed Bandit: Thompson Sampling",
        bandit_cls=Thompson,
        key="thompson",
        log_filename="thompson_logs.csv",
        params={
            "n_arms": (2, 10, 5),
            "batch": (1, 1000, 100),
            "auto_steps": (1, 1000, 100),
            "auto_delay": (0.1, 2.0, 0.3),
            "initial_alpha": (1, 20, 5),
            "initial_beta": (1, 20, 5),
        },
    )


if __name__ == "__main__":
    main()
