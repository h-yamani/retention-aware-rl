import csv

import numpy as np
import torch

from stable_baselines3 import SAC

from retention_rl.memory import (
    EpisodeRecord,
    ValuableEpisodicMemory,
)
from retention_rl.retention.metrics import (
    retention_deficit,
)
from retention_rl.retention.retention_tracker import (
    RetentionTracker,
)
from retention_rl.retention.stored_action_log_prob import (
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


def make_episode(
    model,
    episode_id=1,
    insert_step=100,
):
    states = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    actions = np.array(
        [
            [0.0],
            [0.1],
        ],
        dtype=np.float32,
    )

    rewards = np.array(
        [1.0, 2.0],
        dtype=np.float32,
    )

    insert_nll = trajectory_nll(
        actor=model.actor,
        states=states[:-1],
        actions=actions,
    )

    return EpisodeRecord(
        episode_id=episode_id,
        states=states,
        actions=actions,
        rewards=rewards,
        episode_return=3.0,
        insert_step=insert_step,
        insert_nll=insert_nll,
    )


def test_unchanged_actor_has_zero_retention_deficit():
    model = make_model()

    vem = ValuableEpisodicMemory(
        capacity=5
    )

    episode = make_episode(
        model
    )

    vem.add(
        episode
    )

    tracker = RetentionTracker()

    rows = tracker.evaluate(
        actor=model.actor,
        vem=vem,
        checkpoint_step=1000,
    )

    assert len(rows) == 1

    row = rows[0]

    assert np.isclose(
        row["current_nll"],
        row["insert_nll"],
        atol=1e-7,
    )

    assert np.isclose(
        row["retention_deficit"],
        0.0,
        atol=1e-7,
    )

    assert row[
        "episode_age_steps"
    ] == 900


def test_tracker_updates_episode_metadata():
    model = make_model()

    vem = ValuableEpisodicMemory(
        capacity=5
    )

    episode = make_episode(
        model
    )

    vem.add(
        episode
    )

    tracker = RetentionTracker()

    rows = tracker.evaluate(
        actor=model.actor,
        vem=vem,
        checkpoint_step=1000,
    )

    assert episode.last_nll is not None

    assert np.isclose(
        episode.last_nll,
        rows[0]["current_nll"],
    )

    assert np.isclose(
        episode.last_retention_deficit,
        rows[0]["retention_deficit"],
    )


def test_retention_deficit_matches_metric_after_policy_change():
    model = make_model()

    vem = ValuableEpisodicMemory(
        capacity=5
    )

    episode = make_episode(
        model
    )

    vem.add(
        episode
    )

    with torch.no_grad():
        model.actor.mu.bias.add_(
            2.0
        )

    tracker = RetentionTracker()

    rows = tracker.evaluate(
        actor=model.actor,
        vem=vem,
        checkpoint_step=2000,
    )

    row = rows[0]

    expected = retention_deficit(
        insert_nll=row["insert_nll"],
        current_nll=row["current_nll"],
    )

    assert np.isfinite(
        row["current_nll"]
    )

    assert np.isclose(
        row["retention_deficit"],
        expected,
    )


def test_tracker_writes_csv(tmp_path):
    model = make_model()

    vem = ValuableEpisodicMemory(
        capacity=5
    )

    vem.add(
        make_episode(
            model
        )
    )

    output_csv = (
        tmp_path
        / "retention_history.csv"
    )

    tracker = RetentionTracker(
        output_csv=output_csv
    )

    tracker.evaluate(
        actor=model.actor,
        vem=vem,
        checkpoint_step=1000,
    )

    tracker.evaluate(
        actor=model.actor,
        vem=vem,
        checkpoint_step=2000,
    )

    assert output_csv.exists()

    with output_csv.open(
        "r",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert len(rows) == 2

    assert (
        rows[0]["checkpoint_step"]
        == "1000"
    )

    assert (
        rows[1]["checkpoint_step"]
        == "2000"
    )


def test_empty_vem_is_safe(tmp_path):
    model = make_model()

    vem = ValuableEpisodicMemory(
        capacity=5
    )

    output_csv = (
        tmp_path
        / "retention_history.csv"
    )

    tracker = RetentionTracker(
        output_csv=output_csv
    )

    rows = tracker.evaluate(
        actor=model.actor,
        vem=vem,
        checkpoint_step=1000,
    )

    assert rows == []

    assert not output_csv.exists()
