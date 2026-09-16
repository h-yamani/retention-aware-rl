import numpy as np
import torch

from stable_baselines3 import SAC

from retention_rl.retention.stored_action_log_prob import (
    stored_action_log_prob,
    trajectory_nll,
)


def make_model():
    return SAC(
        policy="MlpPolicy",
        env="Pendulum-v1",
        learning_starts=10,
        buffer_size=100,
        batch_size=8,
        policy_kwargs=dict(
            net_arch=[32, 32],
        ),
        seed=123,
        device="cpu",
        verbose=0,
    )


def test_stored_action_log_prob_shape_and_finite():
    model = make_model()

    obs = np.zeros(
        (4, 3),
        dtype=np.float32,
    )

    actions = np.zeros(
        (4, 1),
        dtype=np.float32,
    )

    log_prob = stored_action_log_prob(
        model.actor,
        obs,
        actions,
    )

    assert log_prob.shape == (4,)

    assert torch.isfinite(
        log_prob
    ).all()


def test_trajectory_nll_is_deterministic():
    model = make_model()

    rng = np.random.default_rng(123)

    states = rng.normal(
        size=(20, 3)
    ).astype(np.float32)

    actions = rng.uniform(
        low=-0.8,
        high=0.8,
        size=(20, 1),
    ).astype(np.float32)

    nll_1 = trajectory_nll(
        model.actor,
        states,
        actions,
    )

    nll_2 = trajectory_nll(
        model.actor,
        states,
        actions,
    )

    assert np.isfinite(nll_1)

    assert np.isclose(
        nll_1,
        nll_2,
        atol=1e-7,
    )


def test_near_boundary_actions_are_finite():
    model = make_model()

    states = np.zeros(
        (4, 3),
        dtype=np.float32,
    )

    actions = np.array(
        [
            [-1.0],
            [-0.999999],
            [0.999999],
            [1.0],
        ],
        dtype=np.float32,
    )

    log_prob = stored_action_log_prob(
        model.actor,
        states,
        actions,
    )

    assert torch.isfinite(
        log_prob
    ).all()


def test_policy_change_changes_stored_action_nll():
    model = make_model()

    rng = np.random.default_rng(321)

    states = rng.normal(
        size=(32, 3)
    ).astype(np.float32)

    actions = np.zeros(
        (32, 1),
        dtype=np.float32,
    )

    before = trajectory_nll(
        model.actor,
        states,
        actions,
    )

    with torch.no_grad():
        model.actor.mu.bias.add_(2.0)

    after = trajectory_nll(
        model.actor,
        states,
        actions,
    )

    assert np.isfinite(before)
    assert np.isfinite(after)

    assert not np.isclose(
        before,
        after,
        atol=1e-5,
    )
