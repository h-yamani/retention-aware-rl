# Retention-Aware Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20RL-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![SAC](https://img.shields.io/badge/Algorithm-SAC-6F42C1)](#reinforcement-learning-backbone)
[![TD3](https://img.shields.io/badge/Algorithm-TD3-6F42C1)](#reinforcement-learning-backbone)
[![MuJoCo](https://img.shields.io/badge/Benchmark-MuJoCo-00599C)](https://mujoco.org/)
[![Research](https://img.shields.io/badge/Research-Retention--Aware%20RL-8A2BE2)](#research-question)
[![Status](https://img.shields.io/badge/Status-Active%20Research-orange)](#development-roadmap)

**Learning a useful behaviour is not enough — an RL agent must also retain it.**

This repository investigates **retention-aware reinforcement learning**: detecting when a continuously learning policy begins to lose support for previously successful behaviours, measuring whether that change corresponds to a genuine loss of capability, and using the resulting signal to decide which experiences should receive additional learning attention.

The project studies two complementary mechanisms:

**RAER — Retention-Aware Active Episode Repetition**
Re-executes valuable behaviours in the environment when evidence suggests that the current policy is losing them.

**RPER — Retention-Aware Prioritized Experience Replay**
Prioritizes stored experiences associated with retention loss during replay-buffer optimization.

---

# Motivation

Deep reinforcement learning agents continually update their policies as new experience arrives.

This creates an important question:

> **Can an agent discover a valuable behaviour, improve elsewhere, and later lose its ability to reproduce that behaviour?**

Standard replay mechanisms prioritize experience using signals such as temporal-difference error, reward, novelty, or recency.

However, these signals do not directly measure whether the **current policy is moving away from a behaviour that was previously successful**.

This project introduces a different perspective:

```text
Discover valuable behaviour
        ↓
Store behavioural reference
        ↓
Policy continues learning
        ↓
Measure current support for stored behaviour
        ↓
Has support declined?
        ↓
Test whether capability has also declined
        ↓
Retention-aware intervention
       / \
      /   \
   RAER   RPER
 active  passive
```

The central idea is simple:

> **Learning systems should not only ask what experience is important to learn from, but also what valuable behaviour they are beginning to lose.**

---

# Research Question

The first scientific question investigated by this repository is:

> **When an RL agent discovers a high-return behaviour, does that behaviour later lose policy support, and does this loss predict deterioration in the agent's ability to reproduce the behaviour?**

The project intentionally separates two concepts:

### Policy-support decay

The current policy becomes less likely to generate actions associated with a previously successful trajectory.

### Capability loss

When returned to the same initial condition, the current policy performs worse than when the valuable behaviour was originally discovered.

A decrease in policy support alone is **not automatically forgetting**.

The agent may simply have discovered a better behaviour.

Therefore, retention signals are validated against an independent capability test before being used as learning priorities.

---

# Core Retention Signal

Consider a valuable trajectory

```math
\tau = \{(s_1,a_1), \ldots, (s_T,a_T)\}.
```

For a stochastic SAC policy, the current negative log-likelihood of the stored actions is

```math
\mathrm{NLL}_t(\tau)
=
-\frac{1}{T}
\sum_{j=1}^{T}
\log \pi_{\theta_t}(a_j|s_j).
```

When the trajectory is first stored, its policy compatibility is recorded:

```math
\mathrm{NLL}_{store}(\tau).
```

Later, the change in policy support is measured by

```math
\Delta \mathrm{NLL}_t(\tau)
=
\mathrm{NLL}_t(\tau)
-
\mathrm{NLL}_{store}(\tau).
```

The **retention deficit** is

```math
F_t(\tau)
=
[\Delta \mathrm{NLL}_t(\tau)]_+.
```

Interpretation:

```text
F ≈ 0
Current policy still supports the stored behaviour.

F > 0
Current policy supports the behaviour less than when it was stored.

Large F
Strong candidate for potential retention loss.
```

---

# Capability Validation

Policy-support decay does not necessarily imply harmful forgetting.

To test whether the agent has actually lost capability, the project restores the initial simulator state associated with a stored valuable episode and evaluates the current policy from the same condition.

For a stored trajectory with historical return \(R(\tau)\):

```math
G_t(\tau)
=
R(\tau)
-
\bar{R}^{revisit}_t(\tau)
```

where

```math
\bar{R}^{revisit}_t(\tau)
```

is the current policy's mean return when revisiting the original initial state.

A large positive capability gap means the current agent performs worse from a state where it previously demonstrated strong performance.

The key validation question becomes:

```text
Retention Deficit F
        ↓
Does it predict
        ↓
Capability Gap G ?
```

This relationship is evaluated before introducing retention-aware interventions.

---

# Retention-Aware Active Episode Repetition — RAER

Once the retention signal is validated, valuable episodes can be selected according to their retention deficit:

```math
\tau^*
=
\arg\max_{\tau \in \mathcal{M}_{valuable}}
F_t(\tau).
```

The selected stored behaviour is then **re-executed in the environment**.

```text
High historical value
        +
High retention deficit
        ↓
Retention-aware selection
        ↓
Active episode repetition
        ↓
Fresh environment transitions
        ↓
Standard off-policy learning
```

RAER therefore changes **environment interaction**, rather than only changing replay-buffer sampling.

---

# Retention-Aware Prioritized Experience Replay — RPER

The second intervention studies whether the same retention signal can improve learning without actively repeating behaviour in the environment.

Instead:

```text
Retention deficit
       ↓
Experience priority
       ↓
Replay-buffer sampling
       ↓
Additional gradient updates
```

This produces a direct comparison between:

| Method          | Mechanism                              | Environment Interaction |
| --------------- | -------------------------------------- | ----------------------- |
| Standard Replay | Uniform sampling                       | Standard                |
| TD-PER          | TD-error prioritization                | Standard                |
| RPER            | Retention-aware replay                 | Standard                |
| RAER            | Retention-aware behavioural repetition | Modified                |

This comparison asks an important question:

> **When valuable behaviour begins to fade, is it better to replay its data or to actively perform the behaviour again?**

---

# Valuable Episodic Memory

Retention is measured over a compact **Valuable Episodic Memory (VEM)**.

Each stored episode can contain:

```python
EpisodeRecord(
    episode_id,
    states,
    actions,
    rewards,
    episode_return,
    insert_step,
    insert_nll,
    initial_env_state,
    last_nll,
    last_retention_deficit,
    repeat_count,
)
```

The initial implementation maintains a top-\(M\) memory of high-return policy-generated episodes.

The VEM is deliberately separated from the ordinary transition replay buffer:

```text
Replay Buffer
    │
    └── large transition-level training memory

Valuable Episodic Memory
    │
    └── small behaviour-level retention memory
```

---

# Reinforcement Learning Backbone

The project is designed around off-policy continuous-control reinforcement learning.

Initial development focuses on:

| Algorithm | Policy Type   | Retention Measurement                |
| --------- | ------------- | ------------------------------------ |
| SAC       | Stochastic    | Change in action log-likelihood      |
| TD3       | Deterministic | Change in normalized action mismatch |

**SAC is the first implementation target** because its stochastic policy provides a natural probabilistic measure of behavioural support.

TD3 support is planned after the SAC retention signal has been validated.

---

# Experimental Methodology

The project follows a staged experimental design.

### Stage I — Does the retention signal mean anything?

No intervention is applied.

The agent trains normally while the system records:

```text
historical return
current NLL
insertion NLL
retention deficit
episode age
revisit performance
capability gap
```

Analysis includes:

* Spearman correlation between retention deficit and capability gap
* comparison against absolute NLL
* comparison against episode age
* high-vs-low retention-deficit quartiles
* regression / partial-correlation analysis
* capability-loss classification and AUROC

### Stage II — Can active repetition repair retention loss?

Compare:

```text
Base RL
Return-based repetition
Absolute-support repetition
RAER
```

under controlled environment-interaction budgets.

### Stage III — Active versus passive repair

Compare:

```text
Base RL
TD-PER
RPER
RAER
```

under matched experimental conditions.

---

# Evaluation Metrics

Experiments are designed to report:

* learning curves
* Area Under the Learning Curve (AUC)
* Interquartile Mean (IQM)
* final performance
* steps to performance thresholds
* retention-deficit statistics
* capability-gap statistics
* correlation analysis
* multi-seed uncertainty

The objective is not merely to demonstrate higher return, but to determine whether **retention loss is measurable, predictive, and actionable**.

---

# Repository Structure

```text
retention-aware-rl/
│
├── configs/
│   └── stage1.yaml
│
├── retention_rl/
│   │
│   ├── memory/
│   │   └── valuable_episodic_memory.py
│   │
│   └── retention/
│       └── metrics.py
│
├── scripts/
│   └── demo_retention_signal.py
│
├── tests/
│   ├── test_retention_metrics.py
│   └── test_vem.py
│
├── results/
│
├── README.md
├── requirements.txt
└── .gitignore
```

The structure will expand as SAC integration, environment-state restoration, RAER, RPER, experiment management, and analysis modules are implemented.

---

# Current Implementation

The first implementation milestone contains:

* top-\(M\) Valuable Episodic Memory
* policy-generated episode filtering
* retention-deficit calculation
* capability-gap calculation
* unit tests for VEM behaviour
* unit tests for retention metrics
* Stage-I experiment configuration
* minimal retention-signal demonstration

Importantly, **RAER and RPER are not enabled at this stage**.

The project first validates whether the proposed retention signal corresponds to measurable capability loss.

---

# Development Roadmap

```text
Stage I — Retention Signal
    ↓
SAC evaluated-action log probability
    ↓
Store behavioural compatibility
    ↓
Periodic retention measurement
    ↓
Simulator-state restoration
    ↓
Capability revisit experiments
    ↓
Statistical validation

Stage II — RAER
    ↓
Retention-aware episode selection
    ↓
Active behavioural repetition
    ↓
Controlled comparison

Stage III — RPER
    ↓
Retention-aware replay priorities
    ↓
Active vs passive comparison
    ↓
Ablation studies
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/h-yamani/retention-aware-rl.git
cd retention-aware-rl
```

Create a Python environment:

```bash
conda create -n retention-rl python=3.10
conda activate retention-rl
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the tests:

```bash
python -m pytest -q
```

---

# Quick Retention-Signal Demo

A minimal example is provided to illustrate the bookkeeping behind the Stage-I signal.

```bash
python scripts/demo_retention_signal.py
```

Example concept:

```text
NLL when behaviour was stored : 1.20
Current NLL                   : 2.05
Retention deficit             : 0.85

Historical episode return     : 900
Current revisit return        : 650
Capability gap                : 250
```

This example is **illustrative only** and is not an experimental result.

---

# Technical Focus

This repository demonstrates work across:

`Reinforcement Learning` · `Deep Learning` · `PyTorch` · `SAC` · `TD3` · `Experience Replay` · `Prioritized Replay` · `Continuous Control` · `MuJoCo` · `Experimental Design` · `Statistical Evaluation` · `Scientific Computing` · `Robotic Learning`

---

# Research Context

This project builds on previous work investigating how deliberate behavioural repetition can improve sample efficiency in reinforcement learning.

The current research extends that direction by moving from:

```text
"What successful behaviour should be repeated?"
```

toward:

```text
"What valuable behaviour is the agent beginning to lose?"
```

This shifts repetition from a reward-driven mechanism toward a **retention-aware learning mechanism**.

---

# Status

🚧 **Active research and development**

The repository is being developed incrementally so that each research claim is validated before the corresponding intervention is introduced.

Current priority:

> **Validate whether policy-support decay predicts measurable loss of previously demonstrated capability.**

Experimental results will be added as the validation stages are completed.

---

# Author

**Hoda Yamani**

Machine Learning Engineer · Reinforcement Learning Researcher · Applied AI

Research interests include reinforcement learning, robot learning, experience replay, sample-efficient learning, continuous control, and intelligent autonomous systems.

---

# License

MIT License
