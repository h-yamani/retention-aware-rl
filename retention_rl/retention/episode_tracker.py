from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from retention_rl.memory import (
    EpisodeRecord,
    ValuableEpisodicMemory,
)
from retention_rl.retention.stored_action_log_prob import (
    trajectory_nll,
)


@dataclass
class EpisodeTracker:
    """
    Collect one complete environment episode for Stage-I retention.

    Storage convention
    ------------------
    For an episode with T transitions:

        states  = [s_0, s_1, ..., s_T]   -> length T + 1
        actions = [a_0, a_1, ..., a_T-1] -> length T
        rewards = [r_0, r_1, ..., r_T-1] -> length T

    Therefore retention likelihood is evaluated using:

        states[:-1], actions

    so each historical action a_t is evaluated at the state s_t
    where that action was originally taken.

    Stage I only admits episodes whose actions were generated
    entirely by the learned SAC policy.
    """

    episode_id: int

    states: list[np.ndarray] = field(
        default_factory=list
    )

    actions: list[np.ndarray] = field(
        default_factory=list
    )

    rewards: list[float] = field(
        default_factory=list
    )

    policy_generated: bool = True

    def start(
        self,
        initial_state,
    ) -> None:
        """
        Start an episode from s_0.
        """

        self.states = [
            np.asarray(
                initial_state,
                dtype=np.float32,
            ).copy()
        ]

        self.actions = []
        self.rewards = []

        self.policy_generated = True

    def add_transition(
        self,
        action,
        reward: float,
        next_state,
        action_from_policy: bool,
    ) -> None:
        """
        Add:

            (a_t, r_t, s_{t+1})

        to the current episode.
        """

        if len(self.states) == 0:
            raise RuntimeError(
                "EpisodeTracker.start() must be called "
                "before adding transitions."
            )

        self.actions.append(
            np.asarray(
                action,
                dtype=np.float32,
            ).copy()
        )

        self.rewards.append(
            float(reward)
        )

        self.states.append(
            np.asarray(
                next_state,
                dtype=np.float32,
            ).copy()
        )

        if not action_from_policy:
            self.policy_generated = False

    @property
    def episode_return(
        self,
    ) -> float:
        return float(
            np.sum(
                self.rewards,
                dtype=np.float64,
            )
        )

    @property
    def length(
        self,
    ) -> int:
        return len(
            self.actions
        )

    def finalize(
        self,
        actor,
        insert_step: int,
    ) -> EpisodeRecord | None:
        """
        Finalize the episode.

        Returns
        -------
        EpisodeRecord
            For a fully policy-generated episode.

        None
            If any action in the episode came from random
            warmup/external intervention.

        IMPORTANT
        ---------
        insert_nll is measured using the exact historical
        actions taken during this episode.
        """

        if self.length == 0:
            raise RuntimeError(
                "Cannot finalize an empty episode."
            )

        if not self.policy_generated:
            return None

        states = np.asarray(
            self.states,
            dtype=np.float32,
        )

        actions = np.asarray(
            self.actions,
            dtype=np.float32,
        )

        rewards = np.asarray(
            self.rewards,
            dtype=np.float32,
        )

        if len(states) != len(actions) + 1:
            raise RuntimeError(
                "Expected T+1 states for T actions."
            )

        insert_nll = trajectory_nll(
            actor=actor,
            states=states[:-1],
            actions=actions,
        )

        return EpisodeRecord(
            episode_id=int(
                self.episode_id
            ),

            states=states,

            actions=actions,

            rewards=rewards,

            episode_return=self.episode_return,

            insert_step=int(
                insert_step
            ),

            insert_nll=float(
                insert_nll
            ),

            source="policy",
        )


def finalize_and_add_to_vem(
    tracker: EpisodeTracker,
    actor,
    insert_step: int,
    vem: ValuableEpisodicMemory,
) -> tuple[EpisodeRecord | None, bool]:
    """
    Finalize a tracked episode and attempt VEM admission.

    Returns
    -------
    episode:
        The completed EpisodeRecord, or None if the episode
        was not fully policy-generated.

    retained:
        True only if the completed episode remains in VEM
        after top-M admission.
    """

    episode = tracker.finalize(
        actor=actor,
        insert_step=insert_step,
    )

    if episode is None:
        return None, False

    retained = vem.add(
        episode
    )

    return episode, retained
