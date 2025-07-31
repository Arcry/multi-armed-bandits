from typing import List, Optional

import numpy as np
import random


class BaseBandit:
    """
    Base class for multi-armed bandits.
    Implements common update, reward, and rmse routines.
    """

    def __init__(self, n_arms: int, true_probs: Optional[List[float]] = None):
        self.n_arms = n_arms
        self.true_probs = (
            true_probs or np.random.uniform(0.1, 0.9, size=n_arms).tolist()
        )
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms

    def reward(self, arm: int) -> int:
        """
        Simulate binary reward: 1 with probability true_probs[arm], else 0.
        """
        return int(random.random() < self.true_probs[arm])

    def update(self, arm: int, reward: int) -> None:
        """
        Incremental update of estimated mean: new = old + (r - old)/count
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def estimated_means(self) -> List[float]:
        """
        Default estimated means (just current values).
        Override in subclasses if needed.
        """
        return self.values

    def rmse(self) -> float:
        """
        RMSE between estimated means and true probabilities.
        """
        errors = [
            (est - true) ** 2
            for est, true in zip(self.estimated_means(), self.true_probs)
        ]
        return float(np.sqrt(np.mean(errors)))

    def pull(self) -> int:
        """
        Abstract pull method to select an arm. Must be overridden.
        """
        raise NotImplementedError
