from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

from retention_rl.envs.mujoco_state import (
    capture_mujoco_state,
)
from retention_rl.memory import (
    ValuableEpisodicMemory,
)
from retention_rl.retention.capability_evaluator import (
    CapabilityEvaluator,
)
from retention_rl.retention.episode_tracker import (
    EpisodeTracker,
    finalize_and_add_to_vem,
)
from retention_rl.retention.retention_tracker import (
    RetentionTracker,
)


class Stage1RetentionCallback(BaseCallback):
    """
    Stage-I observational retention callback.

    Responsibilities:
      1. Track complete episodes.
      2. Reject episodes containing SAC warm-up actions.
      3. Capture the exact MuJoCo state at episode start.
      4. Store valuable policy-generated episodes in VEM.
      5. Measure trajectory retention deficit F periodically.
      6. Prepare an independent environment for later capability
         revisit measurements.

    IMPORTANT:
        This callback does not modify SAC actions, gradients,
        replay-buffer sampling, rewards, or policy updates.
    """

    def __init__(
        self,
        output_dir,
        vem_capacity: int = 20,
        retention_interval_steps: int = 10_000,
        learning_starts: int = 10_000,
        revisit_rollouts: int = 5,
        verbose: int = 1,
    ):
        super().__init__(
            verbose=verbose
        )

        if vem_capacity <= 0:
            raise ValueError(
                "vem_capacity must be positive."
            )

        if retention_interval_steps <= 0:
            raise ValueError(
                "retention_interval_steps must be positive."
            )

        if learning_starts < 0:
            raise ValueError(
                "learning_starts must be non-negative."
            )

        if revisit_rollouts <= 0:
            raise ValueError(
                "revisit_rollouts must be positive."
            )

        self.output_dir = Path(
            output_dir
        )

        self.vem_capacity = int(
            vem_capacity
        )

        self.retention_interval_steps = int(
            retention_interval_steps
        )

        self.learning_starts = int(
            learning_starts
        )

        self.revisit_rollouts = int(
            revisit_rollouts
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

        self.capability_evaluator = None
        self.revisit_env = None

        self.episode_tracker = None

        self.next_episode_id = 0

        self.next_retention_step = (
            self.retention_interval_steps
        )

        self.total_completed_episodes = 0
        self.policy_completed_episodes = 0
        self.vem_admissions = 0

    def _get_training_env(self):
        """
        Return the single underlying Gymnasium training environment.

        Stage I currently requires one environment.
        """

        vec_env = self.model.get_env()

        if vec_env.num_envs != 1:
            raise RuntimeError(
                "Stage1RetentionCallback currently "
                "supports exactly one environment."
            )

        if not hasattr(vec_env, "envs"):
            raise RuntimeError(
                "Training VecEnv does not expose envs."
            )

        return vec_env.envs[0]

    def _get_env_id(self) -> str:
        """
        Recover the Gymnasium environment ID from the training env.
        """

        training_env = self._get_training_env()

        base_env = training_env.unwrapped

        if base_env.spec is None:
            raise RuntimeError(
                "Could not determine environment specification."
            )

        if base_env.spec.id is None:
            raise RuntimeError(
                "Could not determine environment ID."
            )

        return str(
            base_env.spec.id
        )

    def _on_training_start(self):
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.model.get_env().num_envs != 1:
            raise RuntimeError(
                "Stage1RetentionCallback currently "
                "supports exactly one environment."
            )

        # -----------------------------------------------------
        # Independent environment for capability revisits.
        #
        # This environment is never used for SAC training.
        # -----------------------------------------------------

        env_id = self._get_env_id()

        self.revisit_env = gym.make(
            env_id
        )

        self.capability_evaluator = (
            CapabilityEvaluator(
                env=self.revisit_env,
                n_rollouts=self.revisit_rollouts,
                deterministic=True,
            )
        )

        # -----------------------------------------------------
        # Start tracking the current training episode.
        # -----------------------------------------------------

        initial_obs = self.model._last_obs

        if initial_obs is None:
            raise RuntimeError(
                "SAC does not expose an initial observation."
            )

        self._start_new_episode(
            initial_obs[0]
        )

        if self.verbose:
            print(
                "[Stage I] retention measurement initialized"
            )
            print(
                f"[Stage I] VEM capacity={self.vem_capacity}"
            )
            print(
                "[Stage I] retention interval="
                f"{self.retention_interval_steps}"
            )
            print(
                "[Stage I] learning starts="
                f"{self.learning_starts}"
            )
            print(
                "[Stage I] revisit rollouts="
                f"{self.revisit_rollouts}"
            )
            print(
                "[Stage I] intervention=DISABLED"
            )

    def _start_new_episode(
        self,
        initial_obs,
    ):
        """
        Begin tracking a new episode and capture its exact
        initial MuJoCo simulator state.
        """

        training_env = self._get_training_env()

        initial_env_state = capture_mujoco_state(
            training_env
        )

        tracker = EpisodeTracker(
            episode_id=self.next_episode_id
        )

        tracker.start(
            np.asarray(
                initial_obs,
                dtype=np.float32,
            ),
            initial_env_state=initial_env_state,
        )

        self.episode_tracker = tracker

        self.next_episode_id += 1

    def _on_step(self) -> bool:
        if self.episode_tracker is None:
            raise RuntimeError(
                "Episode tracker has not been initialized."
            )

        buffer_actions = self.locals.get(
            "buffer_actions"
        )

        rewards = self.locals.get(
            "rewards"
        )

        dones = self.locals.get(
            "dones"
        )

        infos = self.locals.get(
            "infos"
        )

        new_obs = self.locals.get(
            "new_obs"
        )

        if buffer_actions is None:
            raise RuntimeError(
                "SB3 callback locals do not contain "
                "buffer_actions."
            )

        if rewards is None:
            raise RuntimeError(
                "SB3 callback locals do not contain rewards."
            )

        if dones is None:
            raise RuntimeError(
                "SB3 callback locals do not contain dones."
            )

        if infos is None:
            raise RuntimeError(
                "SB3 callback locals do not contain infos."
            )

        if new_obs is None:
            raise RuntimeError(
                "SB3 callback locals do not contain new_obs."
            )

        action = np.asarray(
            buffer_actions[0],
            dtype=np.float32,
        ).copy()

        reward = float(
            rewards[0]
        )

        done = bool(
            dones[0]
        )

        info = infos[0]

        # -----------------------------------------------------
        # Warm-up boundary
        #
        # Callback occurs after num_timesteps increments.
        #
        # Therefore:
        #   callback step <= learning_starts
        #       -> action was random
        #
        #   callback step > learning_starts
        #       -> action came from policy
        # -----------------------------------------------------

        action_from_policy = (
            self.model.num_timesteps
            > self.learning_starts
        )

        # -----------------------------------------------------
        # Preserve true terminal observation.
        #
        # SB3 VecEnv new_obs is already the reset observation
        # when done=True.
        # -----------------------------------------------------

        if done:
            terminal_obs = info.get(
                "terminal_observation"
            )

            if terminal_obs is None:
                raise RuntimeError(
                    "Done transition does not contain "
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

        # -----------------------------------------------------
        # Episode completion
        # -----------------------------------------------------

        if done:
            self._finish_episode()

            # new_obs[0] is now the reset observation for the
            # next episode, and the physical training env has
            # already been reset to that same state.
            self._start_new_episode(
                new_obs[0]
            )

        # -----------------------------------------------------
        # Observational retention measurement
        # -----------------------------------------------------

        self._maybe_measure_retention()

        return True

    def _finish_episode(self):
        self.total_completed_episodes += 1

        episode, retained = (
            finalize_and_add_to_vem(
                tracker=self.episode_tracker,
                actor=self.model.actor,
                insert_step=self.model.num_timesteps,
                vem=self.vem,
            )
        )

        # Episode contained at least one random warm-up action.
        if episode is None:
            if self.verbose:
                print(
                    "[Retention] "
                    f"episode={self.episode_tracker.episode_id} "
                    "skipped "
                    "(contains warm-up/random action)"
                )

            return

        self.policy_completed_episodes += 1

        if retained:
            self.vem_admissions += 1

        if self.verbose:
            print(
                "[Retention] "
                f"episode={episode.episode_id} "
                f"step={episode.insert_step} "
                f"return={episode.episode_return:.3f} "
                f"insert_nll={episode.insert_nll:.6f} "
                f"retained={retained} "
                f"vem_size={len(self.vem)}"
            )

    def _maybe_measure_retention(self):
        while (
            self.model.num_timesteps
            >= self.next_retention_step
        ):
            checkpoint_step = int(
                self.next_retention_step
            )

            rows = self.retention_tracker.evaluate(
                actor=self.model.actor,
                vem=self.vem,
                checkpoint_step=checkpoint_step, 
                capability_evaluator=self.capability_evaluator,
                model=self.model,
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

                    gaps = np.asarray(
                        [
                           row["capability_gap"]
                           for row in rows
                        ],
                        dtype=np.float64,
                    )

                    print(
                        "[Retention checkpoint] "
                        f"step={checkpoint_step} "
                        f"vem={len(self.vem)} "
                        f"F_mean={np.mean(deficits):.6f} "
                        f"F_max={np.max(deficits):.6f} "
                        f"G_mean={np.mean(gaps):.3f} "
                        f"G_max={np.max(gaps):.3f}"
                    )

                else:
                    print(
                        "[Retention checkpoint] "
                        f"step={checkpoint_step} "
                        "vem=0"
                    )

            self.next_retention_step += (
                self.retention_interval_steps
            )

    def _on_training_end(self):
        if self.revisit_env is not None:
            self.revisit_env.close()
            self.revisit_env = None
