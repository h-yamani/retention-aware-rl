# Retention-Aware Reinforcement Learning

Research code for validating whether valuable behaviours discovered by an RL agent
lose support under the evolving policy, and whether that support decline predicts
a real loss of behavioural capability.

## Stage I — Validate the retention signal first

This repository intentionally starts with **signal validation only**.

For SAC, a stored valuable episode \(\tau\) has:

- insertion-time negative log-likelihood: `NLL_store`
- current negative log-likelihood: `NLL_now`
- retention deficit:

```text
F(tau) = max(0, NLL_now - NLL_store)
```

The key scientific question is whether larger retention deficit predicts a
larger capability gap when the current policy is evaluated again from the same
initial condition.

Only after this signal is validated should the project add:

1. Retention-Aware Episode Repetition (RAER)
2. Retention-Aware Prioritized Experience Replay (RPER)
3. Active-vs-passive comparisons

## Initial repository structure

```text
retention-aware-rl/
├── configs/
├── retention_rl/
│   ├── memory/
│   │   └── valuable_episodic_memory.py
│   ├── retention/
│   │   └── metrics.py
│   └── utils/
├── scripts/
│   └── demo_retention_signal.py
├── tests/
│   ├── test_retention_metrics.py
│   └── test_vem.py
└── results/
```

## First implementation milestone

Before integrating a full SAC learner, verify that:

1. high-return episodes can be inserted into a bounded VEM;
2. lower-return episodes are rejected when the VEM is full;
3. `retention_deficit = max(0, current_nll - insert_nll)`;
4. no algorithmic repetition is triggered yet.

## Run tests

```bash
python -m pytest -q
```

## Run the tiny signal demo

```bash
python scripts/demo_retention_signal.py
```

This demo does **not** claim an RL result. It only checks the retention-score
calculation and VEM bookkeeping.
