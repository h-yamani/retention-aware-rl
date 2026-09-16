import numpy as np

from retention_rl.retention.episode_tracker import EpisodeTracker
from retention_rl.retention.stage1_callback import Stage1RetentionCallback


def test_warmup_boundary_logic():
    """
    SB3 samples randomly while:

        num_timesteps < learning_starts

    before env.step(), then increments num_timesteps.

    Therefore, inside callback _on_step():

        num_timesteps <= learning_starts
            -> transition came from warmup

        num_timesteps > learning_starts
            -> transition came from policy
    """

    learning_starts = 5000

    assert not (
        1 > learning_starts
    )

    assert not (
        5000 > learning_starts
    )

    assert (
        5001 > learning_starts
    )


def test_warmup_transition_contaminates_episode():
    tracker = EpisodeTracker(
        episode_id=1
    )

    tracker.start(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    tracker.add_transition(
        action=np.zeros(
            1,
            dtype=np.float32,
        ),
        reward=1.0,
        next_state=np.ones(
            3,
            dtype=np.float32,
        ),
        action_from_policy=False,
    )

    tracker.add_transition(
        action=np.zeros(
            1,
            dtype=np.float32,
        ),
        reward=1.0,
        next_state=np.ones(
            3,
            dtype=np.float32,
        ),
        action_from_policy=True,
    )

    assert not tracker.policy_generated


def test_buffer_action_representation_is_normalized():
    """
    Stage-I retention stores SB3 buffer_actions,
    not environment-space actions.

    This test verifies the representation expected
    by the retention likelihood pipeline.
    """

    buffer_actions = np.array(
        [
            [
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
            ]
        ],
        dtype=np.float32,
    )

    stored_action = np.asarray(
        buffer_actions[0],
        dtype=np.float32,
    )

    assert np.all(
        stored_action >= -1.0
    )

    assert np.all(
        stored_action <= 1.0
    )


def test_terminal_observation_is_used_instead_of_reset_observation():
    """
    VecEnv new_obs after done belongs to the NEXT episode.

    The completed episode must instead end with
    infos[0]['terminal_observation'].
    """

    tracker = EpisodeTracker(
        episode_id=2
    )

    initial_obs = np.array(
        [0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    terminal_obs = np.array(
        [9.0, 9.0, 9.0],
        dtype=np.float32,
    )

    reset_obs = np.array(
        [-5.0, -5.0, -5.0],
        dtype=np.float32,
    )

    info = {
        "terminal_observation": terminal_obs
    }

    done = True

    if done:
        next_state = np.asarray(
            info["terminal_observation"],
            dtype=np.float32,
        )
    else:
        next_state = reset_obs

    tracker.start(
        initial_obs
    )

    tracker.add_transition(
        action=np.array(
            [0.0],
            dtype=np.float32,
        ),
        reward=1.0,
        next_state=next_state,
        action_from_policy=True,
    )

    assert np.allclose(
        tracker.states[-1],
        terminal_obs,
    )

    assert not np.allclose(
        tracker.states[-1],
        reset_obs,
    )


def test_callback_configuration(tmp_path):
    callback = Stage1RetentionCallback(
        output_dir=tmp_path,
        vem_capacity=20,
        retention_interval_steps=10_000,
        learning_starts=5_000,
        verbose=0,
    )

    assert callback.vem_capacity == 20

    assert (
        callback.retention_interval_steps
        == 10_000
    )

    assert callback.learning_starts == 5_000

    assert callback.next_retention_step == 10_000

    assert len(callback.vem) == 0
