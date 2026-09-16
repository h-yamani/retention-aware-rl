from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from retention_rl.memory import ValuableEpisodicMemory
from retention_rl.retention.episode_tracker import (
    EpisodeTracker,
    finalize_and_add_to_vem,
)
from retention_rl.retention.retention_tracker import RetentionTracker


class Stage1RetentionCallback(BaseCallback):
    """
    Stage-I retention measurement callback for SB3 SAC.

    This callback is observational only.

    It does NOT:
        - modify SAC updates
        - modify replay sampling
        - repeat actions
        - change rewards
        - change policy actions
        - train from VEM

    It performs three jobs:

    1. Collect complete policy-generated episodes.
    2. Maintain top-M Valuable Episodic Memory (VEM).
    3. Periodically recompute current NLL and retention deficit F.

    Important SB3 conventions
    -------------------------
    The callback receives local variables from
    OffPolicyAlgorithm.collect_rollouts().

    Relevant variables include:

        actions
            Environment-space action.

        buffer_actions
            Normalized action in [-1, 1].
            This is the representation stored by SB3 replay
            and the representation expected by the SAC
            squashed Gaussian log-probability calculation.

        new_obs
            Next VecEnv observation. On terminal transitions,
            VecEnv has already reset, so this may be the first
            observation of the NEXT episode.

        infos[i]["terminal_observation"]
            Correct terminal observation for a finished episode.

    Therefore this callback stores buffer_actions and explicitly
    recovers terminal_observation on episode boundaries.
    """

    def __init__(
        self,
        output_dir: str | Path,
        vem_capacity: int = 20,
        retention_interval_steps: int = 10_000,
        learning_starts: int = 5_000,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)

        if vem_capacity <= 0:
            raise ValueError(
                "vem_capacity must be positive."
            )

        if retention_interval_steps <= 0:
            raise ValueError(
                "retention_interval_steps must be positive."
            )

        self.output_dir = Path(output_dir)

        self.vem_capacity = int(vem_capacity)

        self.retention_interval_steps = int(
            retention_interval_steps
        )

        self.learning_starts = int(
            learning_starts
        )

        self.vem = ValuableEpisodicMemory(
            capacity=self.vem_capacity
        )

        self.retention_tracker = RetentionTracker(
            output_csv=(
                self.output_dir
                / "retention_history.csv"
            )
        )

        self.episode_tracker: EpisodeTracker | None = None

        self.next_episode_id = 0

        self.next_retention_step = (
            self.retention_interval_steps
        )

        self.total_completed_episodes = 0
        self.policy_completed_episodes = 0
        self.vem_admissions = 0

    def _on_training_start(self) -> None:
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.training_env.num_envs != 1:
            raise RuntimeError(
                "Stage-I retention measurement currently "
                "requires exactly one environment."
            )

        initial_obs = self.model._last_obs

        if initial_obs is None:
            raise RuntimeError(
                "Model has no initial observation."
            )

        self._start_new_episode(
            initial_obs[0]
        )

    def _start_new_episode(
        self,
        initial_obs,
    ) -> None:
        self.episode_tracker = EpisodeTracker(
            episode_id=self.next_episode_id
        )

        self.next_episode_id += 1

        self.episode_tracker.start(
            initial_state=np.asarray(
                initial_obs,
                dtype=np.float32,
            )
        )

    def _on_step(self) -> bool:
        """
        Called immediately after env.step() and before
        SB3 stores the transition in replay.

        collect_rollouts() has already incremented
        model.num_timesteps at this point.
        """

        if self.episode_tracker is None:
            raise RuntimeError(
                "Episode tracker was not initialized."
            )

        buffer_actions = self.locals[
            "buffer_actions"
        ]

        rewards = self.locals[
            "rewards"
        ]

        dones = self.locals[
            "dones"
        ]

        infos = self.locals[
            "infos"
        ]

        new_obs = self.locals[
            "new_obs"
        ]

        # Single-env Stage I.
        action = np.asarray(
            buffer_actions[0],
            dtype=np.float32,
        )

        reward = float(
            rewards[0]
        )

        done = bool(
            dones[0]
        )

        info = infos[0]

        # num_timesteps was incremented AFTER this action.
        #
        # If num_timesteps <= learning_starts, this transition
        # was selected during SB3 random warmup.
        action_from_policy = (
            self.model.num_timesteps
            > self.learning_starts
        )

        if done:
            terminal_obs = info.get(
                "terminal_observation"
            )

            if terminal_obs is None:
                raise RuntimeError(
                    "Terminal transition did not contain "
                    "terminal_observation."
                )

            next_state = np.asarray(
                terminal_obs,
                dtype=np.float32,
            )

        else:
            next_state = np.asarray(
                new_obs[0],
                dtype=np.float32,
            )

        self.episode_tracker.add_transition(
            action=action,
            reward=reward,
            next_state=next_state,
            action_from_policy=action_from_policy,
        )

        if done:
            self._finish_episode()

            # new_obs is now the automatically reset
            # observation for the next episode.
            self._start_new_episode(
                new_obs[0]
            )

        self._maybe_measure_retention()

        return True

    def _finish_episode(self) -> None:
        assert self.episode_tracker is not None

        self.total_completed_episodes += 1

        episode, retained = (
            finalize_and_add_to_vem(
                tracker=self.episode_tracker,
                actor=self.model.actor,
                insert_step=self.model.num_timesteps,
                vem=self.vem,
            )
        )

        # Warmup-contaminated episode.
        if episode is None:
            if self.verbose:
                print(
                    "[Retention] "
                    f"episode={self.episode_tracker.episode_id} "
                    "skipped (contains warmup action)"
                )

            return

        self.policy_completed_episodes += 1

        if retained:
            self.vem_admissions += 1

        if self.verbose:
            status = (
                "VEM"
                if retained
                else "rejected"
            )

            print(
                "[Retention] "
                f"episode={episode.episode_id} "
                f"step={self.model.num_timesteps} "
                f"return={episode.episode_return:.2f} "
                f"insert_nll={episode.insert_nll:.4f} "
                f"{status} "
                f"vem_size={len(self.vem)}"
            )

    def _maybe_measure_retention(
        self,
    ) -> None:
        """
        Run retention measurement at each configured
        global timestep checkpoint.

        With interval=10,000 this gives:

            10k, 20k, 30k, ...
        """

        while (
            self.model.num_timesteps
            >= self.next_retention_step
        ):
            checkpoint_step = (
                self.next_retention_step
            )

            rows = self.retention_tracker.evaluate(
                actor=self.model.actor,
                vem=self.vem,
                checkpoint_step=checkpoint_step,
            )

            if self.verbose:
                if rows:
                    deficits = np.asarray(
                        [
                            row["retention_deficit"]
                            for row in rows
                        ],
                        dtype=np.float64,
                    )

                    print(
                        "[Retention checkpoint] "
                        f"step={checkpoint_step} "
                        f"vem={len(rows)} "
                        f"F_mean={deficits.mean():.4f} "
                        f"F_max={deficits.max():.4f}"
                    )

                else:
                    print(
                        "[Retention checkpoint] "
                        f"step={checkpoint_step} "
                        "VEM empty"
                    )

            self.next_retention_step += (
                self.retention_interval_steps
            )
