import numpy as np

from stable_baselines3 import SAC

from retention_rl.memory import (
    ValuableEpisodicMemory,
)
from retention_rl.retention.episode_tracker import (
    EpisodeTracker,
    finalize_and_add_to_vem,
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


def test_tracker_stores_t_plus_one_states():
    tracker = EpisodeTracker(
        episode_id=1
    )

    tracker.start(
        np.array(
            [0.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )

    tracker.add_transition(
        action=np.array(
            [0.1],
            dtype=np.float32,
        ),
        reward=1.0,
        next_state=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        action_from_policy=True,
    )

    tracker.add_transition(
        action=np.array(
            [0.2],
            dtype=np.float32,
        ),
        reward=2.0,
        next_state=np.array(
            [2.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        action_from_policy=True,
    )

    assert tracker.length == 2
    assert len(tracker.states) == 3
    assert len(tracker.actions) == 2
    assert len(tracker.rewards) == 2

    assert np.isclose(
        tracker.episode_return,
        3.0,
    )


def test_random_warmup_episode_is_not_admitted():
    model = make_model()

    tracker = EpisodeTracker(
        episode_id=2
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

    episode = tracker.finalize(
        actor=model.actor,
        insert_step=100,
    )

    assert episode is None


def test_policy_episode_gets_insert_nll():
    model = make_model()

    tracker = EpisodeTracker(
        episode_id=3
    )

    tracker.start(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    tracker.add_transition(
        action=np.array(
            [0.0],
            dtype=np.float32,
        ),
        reward=1.0,
        next_state=np.array(
            [0.1, 0.0, 0.0],
            dtype=np.float32,
        ),
        action_from_policy=True,
    )

    tracker.add_transition(
        action=np.array(
            [0.1],
            dtype=np.float32,
        ),
        reward=2.0,
        next_state=np.array(
            [0.2, 0.0, 0.0],
            dtype=np.float32,
        ),
        action_from_policy=True,
    )

    episode = tracker.finalize(
        actor=model.actor,
        insert_step=200,
    )

    assert episode is not None

    assert episode.source == "policy"

    assert episode.states.shape == (
        3,
        3,
    )

    assert episode.actions.shape == (
        2,
        1,
    )

    assert np.isclose(
        episode.episode_return,
        3.0,
    )

    assert np.isfinite(
        episode.insert_nll
    )


def test_policy_episode_can_enter_vem():
    model = make_model()

    vem = ValuableEpisodicMemory(
        capacity=2
    )

    tracker = EpisodeTracker(
        episode_id=4
    )

    tracker.start(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    tracker.add_transition(
        action=np.array(
            [0.0],
            dtype=np.float32,
        ),
        reward=10.0,
        next_state=np.ones(
            3,
            dtype=np.float32,
        ),
        action_from_policy=True,
    )

    episode, retained = (
        finalize_and_add_to_vem(
            tracker=tracker,
            actor=model.actor,
            insert_step=500,
            vem=vem,
        )
    )

    assert episode is not None
    assert retained
    assert len(vem) == 1
    assert vem.best().episode_id == 4
