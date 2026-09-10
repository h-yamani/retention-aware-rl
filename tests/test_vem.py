import numpy as np

from retention_rl.memory import EpisodeRecord, ValuableEpisodicMemory


def make_episode(episode_id: int, episode_return: float) -> EpisodeRecord:
    rewards = np.array([episode_return / 2, episode_return / 2])
    return EpisodeRecord(
        episode_id=episode_id,
        states=np.zeros((3, 2)),
        actions=np.zeros((2, 1)),
        rewards=rewards,
        episode_return=episode_return,
        insert_step=episode_id * 100,
        insert_nll=1.0,
    )


def test_vem_keeps_top_m_returns():
    vem = ValuableEpisodicMemory(capacity=2)

    assert vem.add(make_episode(1, 100.0))
    assert vem.add(make_episode(2, 200.0))
    assert vem.add(make_episode(3, 300.0))

    returns = [ep.episode_return for ep in vem]
    assert returns == [300.0, 200.0]


def test_vem_rejects_low_return_when_full():
    vem = ValuableEpisodicMemory(capacity=2)

    vem.add(make_episode(1, 200.0))
    vem.add(make_episode(2, 300.0))

    assert not vem.add(make_episode(3, 100.0))
    assert [ep.episode_return for ep in vem] == [300.0, 200.0]
