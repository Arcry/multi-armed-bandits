from typing import List, Optional

import numpy as np
import random

from src.bandit_core import BaseBandit
from src.run_bandit import log_to_csv


class EpsilonGreedy(BaseBandit):
    """Epsilon-Greedy bandit with exploration rate epsilon."""

    def __init__(
        self, n_arms: int, epsilon: float, true_probs: Optional[List[float]] = None
    ):
        super().__init__(n_arms, true_probs)
        self.epsilon = epsilon
        self.explore_count = 0
        self.exploit_count = 0

    def pull(self) -> int:
        if random.random() > self.epsilon:
            self.exploit_count += 1
            return int(np.argmax(self.values))
        else:
            self.explore_count += 1
            return random.randrange(self.n_arms)

    def update(self, arm: int, reward: int) -> None:
        """
        Update counts and estimated mean for the selected arm.
        """
        super().update(arm, reward)


class UCB1(BaseBandit):
    """UCB1 bandit (Upper Confidence Bound 1)."""

    def __init__(self, n_arms: int, true_probs: Optional[List[float]] = None):
        super().__init__(n_arms, true_probs)
        self.total_pulls = 0

    def pull(self) -> int:
        self.total_pulls += 1
        scores = []
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                scores.append(float("inf"))
            else:
                bonus = np.sqrt(2 * np.log(self.total_pulls) / self.counts[i])
                scores.append(self.values[i] + bonus)
        return int(np.argmax(scores))

    def update(self, arm: int, reward: int) -> None:
        """
        Update counts and estimated mean for the selected arm.
        """
        super().update(arm, reward)


class Thompson(BaseBandit):
    """Thompson Sampling bandit with Beta posterior and optional warm-up."""

    def __init__(
        self,
        n_arms: int,
        initial_alpha: int = 1,
        initial_beta: int = 1,
        true_probs: Optional[List[float]] = None,
    ):
        super().__init__(n_arms, true_probs)
        self.alphas = [initial_alpha] * n_arms
        self.betas = [initial_beta] * n_arms

    def warm_up(self, session_state):
        """
        Perform one pull per arm to seed the Beta posterior.
        """
        for arm in range(self.n_arms):
            rwd = self.reward(arm)
            super().update(arm, rwd)
            self.alphas[arm] += rwd
            self.betas[arm] += 1 - rwd
            session_state["step"] += 1
            session_state["rewards_log"].append((session_state["step"], arm, rwd))
            session_state["rmse_log"].append(self.rmse())
            log_to_csv(
                session_state["step"],
                arm,
                rwd,
                self.counts,
                self.values,
                "thompson_logs.csv",
            )

    def pull(self) -> int:
        """
        Sample theta for each arm from Beta(alpha, beta) and choose the arm with the highest sample.
        """
        samples = [np.random.beta(a, b) for a, b in zip(self.alphas, self.betas)]
        return int(np.argmax(samples))

    def update(self, arm: int, reward: int) -> None:
        """
        Update counts, estimated mean, and Beta posterior parameters for the selected arm.
        """
        super().update(arm, reward)
        if reward:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1
