"""Abstract base policy for Hamilton-Jacobi reach-avoid safety.

Mirrors PyHJ/policy/modelfree/BasePolicy_Annealing_Avoid.py from the reference
(CMU-IntentLab/latent-safety, dino branch).

Key contribution: the modified Bellman equation for safety:

    V(x) = min(l(x),  γ · V(x'))

where l(x) is the immediate safety label (negative = unsafe, positive = safe).
This differs from standard RL where V(x) = r(x) + γ·V(x').

The "min" ensures that *any* unsafe state encountered along a trajectory
dominates the value, regardless of how far in the future it occurs.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class BasePolicyAvoid(ABC, nn.Module):
    """Abstract base class for safety-aware (reach-avoid) policies.

    Subclasses must implement:
        forward(obs)  → action tensor
        learn(batch)  → loss dict
    """

    def __init__(self, gamma: float = 0.0):
        super().__init__()
        self._gamma = gamma

    # ------------------------------------------------------------------
    # Gamma (discount / annealing)
    # ------------------------------------------------------------------

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        assert 0.0 <= value <= 1.0, f"gamma must be in [0, 1], got {value}"
        self._gamma = value

    # ------------------------------------------------------------------
    # Modified Bellman equation
    # ------------------------------------------------------------------

    def safety_bellman_target(
        self,
        rewards: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the reach-avoid Bellman target.

        Standard RL:  target = r + γ·V(s')
        Safety RL:    target = min(r,  γ·V(s'))

        Args:
            rewards:     (N,) immediate safety label / reward
            next_values: (N,) V(s') from target critic
            dones:       (N,) float in {0, 1}; 1 if episode ended
        Returns:
            target: (N,) Bellman target values
        """
        future = (1.0 - dones) * self._gamma * next_values
        if self._gamma == 0.0:
            # Warmup phase: no Bellman recursion, just fit immediate reward
            return rewards
        return torch.minimum(rewards, future)

    # ------------------------------------------------------------------
    # Soft target network update
    # ------------------------------------------------------------------

    @staticmethod
    def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        """Polyak / soft update: θ_target ← τ·θ_source + (1-τ)·θ_target."""
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map observation to action."""

    @abstractmethod
    def learn(self, batch: dict) -> dict:
        """Update networks from a replay buffer batch. Return loss dict."""
