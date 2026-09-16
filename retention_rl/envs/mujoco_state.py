from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MujocoState:
    qpos: np.ndarray
    qvel: np.ndarray


def _unwrap_env(env):
    """
    Unwrap Gymnasium/Monitor wrappers until the base MuJoCo env is reached.
    """
    current = env

    while hasattr(current, "env"):
        current = current.env

    return current


def capture_mujoco_state(env) -> MujocoState:
    """
    Capture the physical MuJoCo simulator state required to restore
    HalfCheetah to the same configuration.
    """
    base_env = _unwrap_env(env)

    if not hasattr(base_env, "data"):
        raise TypeError(
            "Environment does not expose MuJoCo data."
        )

    return MujocoState(
        qpos=np.array(
            base_env.data.qpos,
            dtype=np.float64,
            copy=True,
        ),
        qvel=np.array(
            base_env.data.qvel,
            dtype=np.float64,
            copy=True,
        ),
    )


def restore_mujoco_state(
    env,
    state: MujocoState,
) -> np.ndarray:
    """
    Restore a previously captured MuJoCo state and return the
    corresponding observation.
    """
    base_env = _unwrap_env(env)

    if not hasattr(base_env, "set_state"):
        raise TypeError(
            "Environment does not support set_state()."
        )

    base_env.set_state(
        np.array(
            state.qpos,
            dtype=np.float64,
            copy=True,
        ),
        np.array(
            state.qvel,
            dtype=np.float64,
            copy=True,
        ),
    )

    if not hasattr(base_env, "_get_obs"):
        raise TypeError(
            "Environment does not expose _get_obs()."
        )

    observation = base_env._get_obs()

    return np.asarray(
        observation,
        dtype=np.float32,
    ).copy()
