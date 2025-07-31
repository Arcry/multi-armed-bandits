import os
import logging
import time
from datetime import datetime
from typing import List, Type

import numpy as np
import pandas as pd
import streamlit as st

from src.bandit_core import BaseBandit

# Module-level logger
t_logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


def log_to_csv(
    step: int,
    arm: int,
    reward: int,
    counts: List[int],
    values: List[float],
    filename: str,
) -> None:
    """
    Append a log entry to CSV with timestamp, step, arm, reward, counts, and values.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "arm": arm,
        "reward": reward,
        "counts": counts,
        "values": values,
    }
    df = pd.DataFrame([entry])
    write_header = not os.path.exists(filename)
    df.to_csv(
        filename, mode="w" if write_header else "a", header=write_header, index=False
    )
    t_logger.debug(f"Logged to {filename}: {entry}")


def initialize_state(
    key: str, bandit: "BaseBandit", params: dict, filename: str
) -> "BaseBandit":
    """
    Initialize Streamlit session for any bandit:
    - key: session_state key for bandit
    - bandit: instance of BaseBandit
    - params: sidebar parameters (including algorithm-specific ones)
    - filename: log file path
    """
    st.session_state.update({**params, "step": 0, "rewards_log": [], "rmse_log": []})
    # Warm-up if the bandit supports it
    if hasattr(bandit, "warm_up"):
        bandit.warm_up(st.session_state)
    st.session_state[key] = bandit
    return bandit


def run_bandit_app(
    title: str, bandit_cls: Type[BaseBandit], key: str, log_filename: str, params: dict
) -> None:
    """
    Generic Streamlit entrypoint for any bandit algorithm.

    :param title: Page title
    :param bandit_cls: Bandit class (EpsilonGreedy, UCB1, Thompson)
    :param key: Session state key for the bandit instance
    :param log_filename: CSV log file path
    :param params: Dict of sidebar parameters:
                   keys 'n_arms','batch','auto_steps','auto_delay'
                   and algorithm-specific 'epsilon' or 'initial_alpha'
    """

    # Sidebar controls
    st.sidebar.title(f"🛠 {title} Parameters")
    n_arms = st.sidebar.slider("Number of arms", *params["n_arms"])
    batch = st.sidebar.number_input("Batch pulls", *params["batch"])
    auto_steps = st.sidebar.number_input("Auto-run steps", *params["auto_steps"])
    auto_run = st.sidebar.checkbox("▶️ Auto-run", key=f"auto_{key}_run")
    auto_delay = st.sidebar.slider("Auto-delay (s)", *params["auto_delay"])
    algo_kwargs = {}
    if "epsilon" in params:
        algo_kwargs["epsilon"] = st.sidebar.slider("Epsilon", *params["epsilon"])
    if "initial_alpha" in params:
        algo_kwargs["initial_alpha"] = st.sidebar.number_input(
            "initial_alpha", *params["initial_alpha"]
        )
    if "initial_beta" in params:
        algo_kwargs["initial_beta"] = st.sidebar.number_input(
            "initial_beta", *params["initial_beta"]
        )

    # True reward probabilities
    mode = st.sidebar.radio("🎯 True reward probabilities", ["Random", "Manual"])
    if mode == "Manual":
        if f"{key}_manual_default" not in st.session_state:
            st.session_state[f"{key}_manual_default"] = ", ".join(
                f"{x:.2f}" for x in np.random.uniform(0.1, 0.9, n_arms)
            )
        manual_str = st.sidebar.text_input(
            f"Enter {n_arms} probabilities (comma-separated)",
            value=st.session_state[f"{key}_manual_default"],
            key=f"{key}_manual_str",
        )
        try:
            vals = [
                float(x.strip())
                for x in st.session_state[f"{key}_manual_str"].split(",")
            ]
            if len(vals) == n_arms and all(0 <= v <= 1 for v in vals):
                manual_probs = vals
            else:
                st.sidebar.error("Provide valid [0,1] probabilities.")
        except ValueError:
            st.sidebar.error("Invalid format. Use comma-separated numbers.")
    else:
        manual_str = None
        manual_probs = None

    needs_init = (
        key not in st.session_state
        or st.session_state.get("n_arms") != n_arms
        or st.session_state.get("mode") != mode
        or (
            mode == "Manual"
            and st.session_state.get("manual_str")
            != st.session_state.get(f"{key}_manual_str")
        )
        or any(
            st.session_state.get(param) != algo_kwargs[param] for param in algo_kwargs
        )
    )

    if needs_init:
        bandit = initialize_state(
            key,
            bandit_cls(n_arms, **{**algo_kwargs, "true_probs": manual_probs}),
            {"n_arms": n_arms, "mode": mode, "manual_str": manual_str, **algo_kwargs},
            log_filename,
        )
    else:
        bandit = st.session_state[key]

    # Main layout
    st.title(title)
    col1, col2 = st.columns(2)

    def do_pulls(count: int):
        for _ in range(count):
            arm = bandit.pull()
            rwd = bandit.reward(arm)
            bandit.update(arm, rwd)
            st.session_state.step += 1
            st.session_state.rewards_log.append((st.session_state.step, arm, rwd))
            st.session_state.rmse_log.append(bandit.rmse())
            log_to_csv(
                st.session_state.step,
                arm,
                rwd,
                bandit.counts,
                bandit.values,
                log_filename,
            )

    with col1:
        if st.button("Pull Once"):
            do_pulls(1)
        if st.button(f"Pull ×{batch}"):
            do_pulls(batch)
        if auto_run:
            for _ in range(auto_steps):
                do_pulls(batch)
                time.sleep(auto_delay)
            st.success("Auto-run complete—uncheck to stop.")

        st.markdown("### Rewards Over Time")
        if st.session_state.rewards_log:
            df = pd.DataFrame(
                st.session_state.rewards_log, columns=["step", "arm", "reward"]
            )
            st.scatter_chart(df, x="step", y="reward")

        st.markdown("### RMSE (smoothed)")
        if st.session_state.rmse_log:
            smooth = pd.Series(st.session_state.rmse_log)
            smooth = smooth.rolling(window=2000, min_periods=1).mean()
            st.line_chart(pd.DataFrame({"RMSE": smooth}))

    with col2:
        st.markdown("### True vs Estimated")
        dfp = pd.DataFrame(
            {"True": bandit.true_probs, "Estimated": bandit.estimated_means()}
        )
        dfp.index.name = "Arm"
        st.bar_chart(dfp)
        st.markdown("### Pull Counts")
        if (
            algo_kwargs.get("initial_alpha", 1) > 1
            or algo_kwargs.get("initial_beta", 1) > 1
        ):
            st.json(
                {
                    i: {"alpha": a, "beta": b, "counts": c}
                    for i, (a, b, c) in enumerate(
                        zip(bandit.alphas, bandit.betas, bandit.counts)
                    )
                }
            )
        else:
            st.json({i: c for i, c in enumerate(bandit.counts)})

    if st.button("Reset All"):
        if os.path.exists(log_filename):
            os.remove(log_filename)
        st.session_state.clear()
