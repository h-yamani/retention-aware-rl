"""Bounded Valuable Episodic Memory (VEM).

Stage I deliberately stores only policy-generated episodes. Episodes generated
later by an active repetition controller should not be admitted here unless the
experimental protocol is changed explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass
class EpisodeRecord:
    """A valuable policy-generated episode and its retention metadata."""

    episode_id: int
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_return: float
    insert_step: int
    insert_nll: float
    initial_env_state: Any = None
    last_nll: float | None = None
    last_retention_deficit: float = 0.0
    repeat_count: int = 0
    source: str = "policy"

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states)
        self.actions = np.asarray(self.actions)
        self.rewards = np.asarray(self.rewards)

        if len(self.actions) != len(self.rewards):
            raise ValueError("actions and rewards must have the same length.")

        if self.source != "policy":
            raise ValueError(
                "Stage I VEM accepts policy-generated episodes only."
            )


class ValuableEpisodicMemory:
    """Keep the top-M policy-generated episodes by episodic return."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self._episodes: list[EpisodeRecord] = []

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self) -> Iterable[EpisodeRecord]:
        return iter(self._episodes)

    @property
    def episodes(self) -> tuple[EpisodeRecord, ...]:
        return tuple(self._episodes)

    def add(self, episode: EpisodeRecord) -> bool:
        """Add an episode if it belongs to the current top-M set.

        Returns True when the episode is retained in VEM after the update.
        """

        if episode.source != "policy":
            raise ValueError(
                "Stage I VEM accepts policy-generated episodes only."
            )

        # Replace an existing record with the same episode id.
        self._episodes = [
            e for e in self._episodes if e.episode_id != episode.episode_id
        ]

        if len(self._episodes) < self.capacity:
            self._episodes.append(episode)
            self._sort()
            return True

        worst = min(self._episodes, key=lambda e: e.episode_return)

        if episode.episode_return <= worst.episode_return:
            return False

        self._episodes.remove(worst)
        self._episodes.append(episode)
        self._sort()
        return True

    def _sort(self) -> None:
        self._episodes.sort(
            key=lambda e: e.episode_return,
            reverse=True,
        )

    def best(self) -> EpisodeRecord:
        if not self._episodes:
            raise RuntimeError("VEM is empty.")
        return self._episodes[0]
