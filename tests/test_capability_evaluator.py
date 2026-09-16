import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from retention_rl.envs.mujoco_state import (
    capture_mujoco_state,
)
from retention_rl.memory import EpisodeRecord
from retention_rl.retention.capability_evaluator import (
    CapabilityEvaluator,
)


def make_env(seed):
    env = gym.make("HalfCheetah-v5")
    env = Monitor(env)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    return env


def make_model(env, seed=0):
    return SAC(
        "MlpPolicy",
        env,
        learning_starts=10_000,
        seed=seed,
        verbose=0,
    )


def make_episode(env):
    obs, _ = env.reset(seed=123)

    initial_state = capture_mujoco_state(env)

    states = [obs.copy()]
    actions = []
    rewards = []

    for _ in range(10):
        action = np.zeros(
            env.action_space.shape,
            dtype=np.float32,
        )

        next_obs, reward, terminated, truncated, _ = env.step(
            action
        )

        actions.append(action.copy())
        rewards.append(float(reward))
        states.append(next_obs.copy())

        if terminated or truncated:
            break

    return EpisodeRecord(
        episode_id=1,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        episode_return=float(np.sum(rewards)),
        insert_step=100,
        insert_nll=1.0,
        initial_env_state=initial_state,
        source="policy",
    )


def test_capability_evaluator_returns_finite_values():
    source_env = make_env(123)
    revisit_env = make_env(456)

    model = make_model(
        revisit_env,
        seed=0,
    )

    episode = make_episode(
        source_env
    )

    evaluator = CapabilityEvaluator(
        revisit_env,
        n_rollouts=2,
        deterministic=True,
    )

    result = evaluator.evaluate(
        model,
        episode,
    )

    assert result.episode_id == episode.episode_id

    assert result.revisit_returns.shape == (2,)

    assert np.all(
        np.isfinite(result.revisit_returns)
    )

    assert np.isfinite(
        result.revisit_mean_return
    )

    assert np.isfinite(
        result.capability_gap
    )

    source_env.close()
    revisit_env.close()


def test_deterministic_revisit_is_reproducible():
    source_env = make_env(123)
    revisit_env = make_env(456)

    model = make_model(
        revisit_env,
        seed=0,
    )

    episode = make_episode(
        source_env
    )

    evaluator = CapabilityEvaluator(
        revisit_env,
        n_rollouts=3,
        deterministic=True,
    )

    result = evaluator.evaluate(
        model,
        episode,
    )

    assert np.allclose(
        result.revisit_returns,
        result.revisit_returns[0],
        atol=1e-6,
    )

    source_env.close()
    revisit_env.close()


def test_capability_gap_definition():
    source_env = make_env(123)
    revisit_env = make_env(456)

    model = make_model(
        revisit_env,
        seed=0,
    )

    episode = make_episode(
        source_env
    )

    evaluator = CapabilityEvaluator(
        revisit_env,
        n_rollouts=1,
        deterministic=True,
    )

    result = evaluator.evaluate(
        model,
        episode,
    )

    expected = (
        episode.episode_return
        - result.revisit_mean_return
    )

    assert np.isclose(
        result.capability_gap,
        expected,
    )

    source_env.close()
    revisit_env.close()
