from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class WinnerTracker:
    """Track winner neuron history for diversity and stability rewards."""

    num_neurons: int
    history: List[int] = field(default_factory=list)

    def record(self, winner: int) -> None:
        """Append a new winner index to the history."""
        self.history.append(winner)

    def histogram(self) -> torch.Tensor:
        """Return normalized histogram of winner counts."""
        if len(self.history) == 0:
            return torch.full((self.num_neurons,), 1.0 / self.num_neurons)
        counts = torch.bincount(torch.tensor(self.history), minlength=self.num_neurons).float()
        return counts / counts.sum()

    def last_winner(self) -> int | None:
        """Return the most recent winner if available."""
        return self.history[-1] if self.history else None


def reward_sparse(mean_rate: torch.Tensor, rho_target: float) -> torch.Tensor:
    """Compute sparsity reward encouraging mean rate to match target."""
    return -((mean_rate - rho_target) ** 2)


def reward_diversity(hist: torch.Tensor) -> torch.Tensor:
    """Compute diversity reward based on winner histogram uniformity."""
    uniform = torch.full_like(hist, 1.0 / hist.numel())
    return -((hist - uniform) ** 2).sum()


def reward_stability(current_winner: int, tracker: WinnerTracker) -> torch.Tensor:
    """Assess stability of winner neuron across repeated inputs."""
    last = tracker.last_winner()
    if last is None:
        return torch.tensor(0.0)
    return torch.tensor(1.0 if current_winner == last else -1.0)


def reward_classification(correct: bool, margin: float, beta: float) -> torch.Tensor:
    """Return classification reward mixing accuracy and margin."""
    base = torch.tensor(1.0 if correct else -1.0)
    return base + beta * margin


def reward_mimicry(agent_delta: torch.Tensor, teacher_delta: torch.Tensor) -> torch.Tensor:
    """Return gradient mimicry reward as negative MSE between updates."""
    return -torch.mean((agent_delta - teacher_delta) ** 2)
