from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retention_rl.envs.mujoco_state import (
    MujocoState,
    restore_mujoco_state,
)
from retention_rl.retention.metrics import (
    capability_gap,
)


@dataclass
class CapabilityResult:
    episode_id: int
    stored_return: float
    revisit_returns: np.ndarray
    revisit_mean_return: float
    revisit_std_return: float
    capability_gap: float


class CapabilityEvaluator:
    """
    Evaluate whether the current policy can reproduce the
    performance of a previously valuable episode.

    Each revisit:

        1. resets the independent Gymnasium environment,
        2. restores the exact initial MuJoCo state,
        3. runs the current policy from that state,
        4. records the resulting return.

    The reset before restoration is essential because Gymnasium
    wrappers such as OrderEnforcing and TimeLimit maintain their
    own state outside the MuJoCo simulator state.
    """

    def __init__(
        self,
        env,
        n_rollouts: int = 5,
        deterministic: bool = True,
    ):
        if n_rollouts <= 0:
            raise ValueError(
                "n_rollouts must be positive."
            )

        self.env = env
        self.n_rollouts = int(
            n_rollouts
        )
        self.deterministic = bool(
            deterministic
        )

    def evaluate(
        self,
        model,
        episode,
    ) -> CapabilityResult:
        state = episode.initial_env_state

        if state is None:
            raise ValueError(
                "Episode does not contain initial_env_state."
            )

        if not isinstance(
            state,
            MujocoState,
        ):
            raise TypeError(
                "episode.initial_env_state must be "
                "a MujocoState."
            )

        horizon = len(
            episode.actions
        )

        if horizon <= 0:
            raise ValueError(
                "Cannot revisit an empty episode."
            )

        revisit_returns = []

        for _ in range(
            self.n_rollouts
        ):
            # -------------------------------------------------
            # IMPORTANT:
            #
            # Reset Gymnasium wrapper state first.
            #
            # This resets:
            #   - OrderEnforcing state
            #   - TimeLimit elapsed-step counter
            #
            # We then overwrite the randomly reset MuJoCo
            # qpos/qvel with the exact stored episode state.
            # -------------------------------------------------

            self.env.reset()

            observation = restore_mujoco_state(
                self.env,
                state,
            )

            total_return = 0.0

            for _ in range(
                horizon
            ):
                action, _ = model.predict(
                    observation,
                    deterministic=self.deterministic,
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = self.env.step(
                    action
                )

                total_return += float(
                    reward
                )

                if (
                    terminated
                    or truncated
                ):
                    break

            revisit_returns.append(
                total_return
            )

        revisit_returns = np.asarray(
            revisit_returns,
            dtype=np.float64,
        )

        revisit_mean_return = float(
            np.mean(
                revisit_returns
            )
        )

        revisit_std_return = float(
            np.std(
                revisit_returns
            )
        )

        gap = capability_gap(
            stored_return=episode.episode_return,
            revisit_mean_return=revisit_mean_return,
        )

        return CapabilityResult(
            episode_id=int(
                episode.episode_id
            ),
            stored_return=float(
                episode.episode_return
            ),
            revisit_returns=revisit_returns,
            revisit_mean_return=revisit_mean_return,
            revisit_std_return=revisit_std_return,
            capability_gap=float(
                gap
            ),
        )
