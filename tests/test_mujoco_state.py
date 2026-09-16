import gymnasium as gym
import numpy as np

from stable_baselines3.common.monitor import Monitor

from retention_rl.envs.mujoco_state import (
    capture_mujoco_state,
    restore_mujoco_state,
)


def make_halfcheetah():
    env = gym.make("HalfCheetah-v5")
    env = Monitor(env)
    return env


def test_capture_state_is_independent_copy():
    env = make_halfcheetah()

    env.reset(seed=123)

    state = capture_mujoco_state(env)

    qpos_before = state.qpos.copy()
    qvel_before = state.qvel.copy()

    env.step(env.action_space.sample())

    assert np.allclose(
        state.qpos,
        qpos_before,
    )

    assert np.allclose(
        state.qvel,
        qvel_before,
    )

    env.close()


def test_restore_recovers_original_observation():
    env = make_halfcheetah()

    observation, _ = env.reset(seed=123)

    state = capture_mujoco_state(env)

    for _ in range(10):
        env.step(
            env.action_space.sample()
        )

    restored_observation = restore_mujoco_state(
        env,
        state,
    )

    assert restored_observation.shape == observation.shape

    assert np.allclose(
        restored_observation,
        observation,
        atol=1e-6,
    )

    env.close()


def test_same_state_same_action_same_transition():
    env = make_halfcheetah()

    env.reset(seed=321)

    state = capture_mujoco_state(env)

    action = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )

    obs1, reward1, terminated1, truncated1, _ = env.step(
        action
    )

    restore_mujoco_state(
        env,
        state,
    )

    obs2, reward2, terminated2, truncated2, _ = env.step(
        action
    )

    assert np.allclose(
        obs1,
        obs2,
        atol=1e-6,
    )

    assert np.isclose(
        reward1,
        reward2,
        atol=1e-8,
    )

    assert terminated1 == terminated2
    assert truncated1 == truncated2

    env.close()
