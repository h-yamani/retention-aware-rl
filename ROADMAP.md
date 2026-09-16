# Retention-Aware Reinforcement Learning — Research Roadmap

## Project Goal

This project investigates whether reinforcement-learning agents lose previously
acquired valuable behaviours during continued learning, how that loss can be
measured, and whether selective repetition can preserve useful capabilities.

The central idea is:

> A valuable experience should not be repeated simply because it had high
> reward. It should be repeated when there is evidence that the policy is
> losing support for that behaviour and that this loss is associated with
> deterioration of useful capability.

The project is divided into two major phases:

1. **Stage I — Measurement / Discovery**
2. **Stage II — Retention-Aware Intervention**

Stage I MUST remain intervention-free.


---

# 1. Core Research Questions

## RQ1 — Does SAC move away from previously valuable behaviour?

Measure this using the Retention Deficit:

\[
F_i(t)
=
\max
\left(
0,
NLL_i(t)-NLL_i^{insert}
\right)
\]

where:

- `insert_nll` = NLL of the stored actions when the episode was collected.
- `current_nll` = NLL of the same stored actions under the current policy.
- `F` = loss of policy support for the stored behaviour.

Interpretation:

- `F = 0`: no measured loss of policy support.
- larger `F`: current policy has moved further away from the stored actions.


## RQ2 — Does loss of policy support mean loss of capability?

Measure this using the Capability Gap:

\[
G_i(t)
=
R_i^{stored}
-
R_i^{revisit}(t)
\]

where:

- `stored_return` = return of the original valuable episode.
- `revisit_return` = return of the current policy when restored to the same
  initial simulator state.

Interpretation:

- `G > 0`: current policy performs worse → possible capability loss.
- `G = 0`: capability approximately reproduced.
- `G < 0`: current policy performs better → old behaviour may simply be obsolete.


## RQ3 — Can retention deficit predict FUTURE capability loss?

The most important Stage-I hypothesis is not simply:

\[
F(t) \leftrightarrow G(t)
\]

but:

\[
\boxed{
F_i(t), \Delta F_i(t)
\rightarrow
G_i(t+\Delta)
}
\]

Question:

> Can we detect that a valuable behaviour is becoming unsupported BEFORE
> the agent actually loses the capability?


## RQ4 — Which valuable episode should be repeated?

Candidate signals:

- Episode return: `R`
- Return rank/percentile: `R_rank`
- Entropy at episode insertion: `H_insert`
- Current entropy: `H_current`
- Retention deficit: `F`
- Change in retention deficit: `Delta_F`
- Capability gap: `G`
- Change in capability gap: `Delta_G`
- Episode age

The final RAER trigger should be selected from experimental evidence rather
than assumed in advance.


---

# 2. Frozen SAC Development Protocol

Current environment:

- `HalfCheetah-v5`

Algorithm:

- Stable-Baselines3 SAC
- Uniform replay
- No PER
- No episode repetition
- No retention intervention during Stage I

SAC configuration:

- learning rate: `3e-4`
- replay buffer: `1,000,000`
- learning starts: `10,000`
- batch size: `256`
- tau: `0.005`
- gamma: `0.99`
- train frequency: `1`
- gradient steps: `1`
- entropy coefficient: `auto`
- target entropy: `auto`
- actor/critic hidden layers: `[256, 256]`

Experiment configuration:

- diagnostic frequency: `1,000`
- evaluation frequency: `5,000`
- checkpoint frequency: `10,000`
- retention measurement frequency: `10,000`
- VEM capacity: `20`
- capability revisit rollouts: `5`

IMPORTANT:

Do not change these values during comparison experiments unless the change is
explicitly part of an ablation.


---

# 3. Valuable Episodic Memory (VEM)

VEM stores valuable complete policy-generated episodes.

Current capacity:

`M = 20`

Episodes are ranked by episodic return.

Random SAC warm-up episodes MUST NOT enter VEM.

Each stored EpisodeRecord currently contains:

- episode ID
- states
- actions
- rewards
- episodic return
- insertion step
- insertion NLL
- initial MuJoCo simulator state
- last NLL
- last retention deficit
- repeat count
- source

The initial MuJoCo state is stored so the current policy can later be evaluated
from the same physical starting state.


---

# 4. Completed Work

## DONE — Reproducible SAC baseline

A clean SB3 SAC baseline has been implemented.

Baseline includes:

- deterministic evaluation
- detailed evaluation
- SAC diagnostics
- checkpoints
- metadata
- package versions
- Git commit metadata
- fixed seeds
- TensorBoard/SB3 logging


## DONE — SAC diagnostics

Diagnostics include:

- Q statistics
- target Q
- TD error
- policy log probability
- policy entropy
- actor mean/log standard deviation
- sampled actions
- replay rewards/dones
- entropy coefficient
- gradient norms

Diagnostic RNG state is preserved so diagnostics do not modify training.


## DONE — Stored-action likelihood

Implemented correct likelihood evaluation of STORED actions under the current
SAC policy.

Important:

Do NOT use newly sampled actions for retention measurement.

For each stored pair:

\[
(s_j,a_j)
\]

evaluate:

\[
\log \pi_{\theta_t}(a_j|s_j)
\]

using the current tanh-squashed Gaussian SAC actor.


## DONE — Retention Deficit F

For each VEM episode:

\[
F_i(t)
=
\max
\left(
0,
NLL_i(t)-NLL_i^{insert}
\right)
\]

This measures how much probability support for the original behaviour has
decreased.


## DONE — Exact MuJoCo state capture/restoration

Implemented:

`retention_rl/envs/mujoco_state.py`

The episode's initial `qpos` and `qvel` are captured.

The independent revisit environment can be restored to the same physical
starting state.


## DONE — Capability Gap G

Implemented:

\[
G_i(t)
=
R_i^{stored}
-
R_i^{revisit}(t)
\]

The current policy is evaluated from the stored initial state.

Capability measurements are performed in an independent environment and do
not alter the training environment.


## DONE — End-to-end Stage-I integration

Successful integration run:

- Environment: `HalfCheetah-v5`
- Steps: `21,000`
- Seed: `989`
- Learning starts: `10,000`
- Intervention: OFF

Results:

- 21 completed episodes
- 11 fully policy-generated episodes
- 11 VEM admissions
- VEM size at end: 11

At checkpoint 20k:

- VEM size: 10
- mean retention deficit F: approximately `4.95`
- maximum F: approximately `12.58`
- mean capability gap G: approximately `-80.39`
- maximum G: approximately `+20.77`


---

# 5. Preliminary Finding — DO NOT OVERCLAIM

The 21k integration run produced an important preliminary observation.

Older stored episodes often developed large retention deficits.

Example:

Episode 10:

- insert NLL ≈ 4.10
- current NLL ≈ 16.68
- F ≈ 12.58

Therefore the current policy had moved substantially away from the old action
sequence.

However:

- stored return ≈ -291.61
- revisit return ≈ -142.45
- G ≈ -149.16

The current policy actually performed much better.

Therefore:

\[
\boxed{
\text{loss of support for an old trajectory}
\neq
\text{automatic loss of capability}
}
\]

This is currently only a preliminary observation from one short run and one
retention checkpoint.

It is NOT yet a paper-level conclusion.


---

# 6. Important Scientific Consequence

Do NOT implement the naive rule:

\[
\text{repeat} = \arg\max_i F_i
\]

A high F may simply mean that SAC has found a better behaviour.

Likewise, do NOT automatically repeat:

- the highest-return episode
- the highest-entropy episode
- the oldest episode

without experimental evidence.

The episode-selection mechanism must distinguish:

1. useful capability becoming lost

from

2. obsolete behaviour being replaced by better behaviour.


---

# 7. NEXT STEP — DO THIS FIRST

## Stage-I Measurement Upgrade

Before the 100k observational experiment, add temporal and entropy measurements.

### Add to EpisodeRecord / retention logs

Record:

- `insert_entropy`
- `current_entropy`
- `delta_entropy`
- `previous_F`
- `delta_F`
- `previous_G`
- `delta_G`

Definitions:

\[
\Delta F_i(t)
=
F_i(t)-F_i(t-\Delta)
\]

\[
\Delta G_i(t)
=
G_i(t)-G_i(t-\Delta)
\]

These allow us to study whether retention deterioration predicts later
capability deterioration.


## Capability revisit correction

Current 5 revisit rollouts are deterministic.

In deterministic HalfCheetah, this causes revisit standard deviation to be
approximately zero.

Before the 100k experiment:

- use stochastic SAC policy revisits
- preserve Python RNG
- preserve NumPy RNG
- preserve PyTorch RNG
- restore RNG after capability evaluation

This ensures:

1. the 5 revisit rollouts are meaningful;
2. capability uncertainty can be estimated;
3. capability evaluation does NOT alter SAC training.


---

# 8. NEXT EXPERIMENT — 100k OBSERVATIONAL STAGE I

After the measurement upgrade and tests pass:

Run:

- Environment: HalfCheetah-v5
- Steps: 100,000
- Initial discovery seed: 0
- Learning starts: 10,000
- VEM capacity: 20
- Retention checkpoint: every 10,000
- Revisit rollouts: 5
- Intervention: OFF

This experiment is NOT an RAER experiment.

It is a discovery experiment.


## At every retention checkpoint collect

For every VEM episode:

\[
R_i
\]

\[
H_i^{insert}
\]

\[
H_i^{current}
\]

\[
F_i(t)
\]

\[
\Delta F_i(t)
\]

\[
G_i(t)
\]

\[
\Delta G_i(t)
\]

plus:

- episode age
- VEM rank
- VEM size
- stored return
- revisit mean return
- revisit return standard deviation


---

# 9. Stage-I Analysis

The first analysis asks:

## Does F predict G?

Measure association between:

\[
F(t)
\]

and

\[
G(t)
\]


## More important: does F predict FUTURE G?

Test:

\[
F(t)
\rightarrow G(t+10k)
\]

and:

\[
\Delta F(t)
\rightarrow G(t+10k)
\]


## Does entropy help?

Test whether:

\[
H_{insert}
\]

adds useful predictive information.

Possible hypothesis:

valuable behaviours discovered while the policy is highly exploratory may
become unsupported as SAC entropy decreases.


---

# 10. Predictor Ablation

Compare candidate predictors of future capability loss.

### A1

Return only:

\[
R
\]

### A2

Entropy only:

\[
H
\]

### A3

Retention deficit:

\[
F
\]

### A4

Retention-deficit trend:

\[
\Delta F
\]

### A5

Value + retention:

\[
R + F
\]

### A6

Value + retention trend:

\[
R + \Delta F
\]

### A7

Entropy + retention:

\[
H + F
\]

### A8

Value + entropy + retention:

\[
R + H + F
\]

### A9

Value + retention + trend:

\[
R + F + \Delta F
\]

### A10

Full candidate:

\[
R + H + F + \Delta F
\]


---

# 11. IMPORTANT — Normalize Signals

Do NOT directly add raw:

- reward
- entropy
- F
- Delta F

because they have different numerical scales.

Use VEM-relative percentile/rank values:

\[
\tilde R_i
\]

\[
\tilde H_i
\]

\[
\tilde F_i
\]

\[
\widetilde{\Delta F}_i
\]

For example:

\[
Score_i
=
\tilde R_i+\tilde F_i
\]

For the initial study, use equal weighting.

Do NOT perform a large arbitrary weight search yet.


---

# 12. Candidate RAER Principle

The current conceptual target is:

\[
\boxed{
\text{valuable}
+
\text{retention risk}
+
\text{evidence/prediction of capability deterioration}
}
\]

A valuable episode should become a repetition candidate when the current
policy appears to be losing useful support for it.

Ideally Stage I will show:

\[
F_i(t)
\text{ or }
\Delta F_i(t)
\rightarrow
G_i(t+\Delta)
\]

If this relationship exists, RAER can intervene BEFORE capability is fully
lost.


---

# 13. Stage II — RAER

RAER:

**Retention-Aware Active Episode Repetition**

Only start this stage after Stage-I evidence is analysed.

RAER should:

1. maintain VEM;
2. measure retention risk;
3. identify valuable episodes at risk;
4. selectively repeat an episode;
5. avoid repeating obsolete trajectories unnecessarily.


---

# 14. RAER Episode-Selection Ablation

Initial intervention comparison:

### SAC

No repetition.

### Random-IER

Random episode from VEM.

Purpose:

Does repetition itself help?


### Return-IER

Repeat highest-value episode.

Purpose:

Is reward alone sufficient?


### F-IER

Repeat highest-retention-deficit episode.

Purpose:

Is retention alone sufficient?


### Return + Entropy

Select based on:

\[
\tilde R+\tilde H
\]

Purpose:

Test the high-value/high-entropy hypothesis.


### Return + F

Select based on:

\[
\tilde R+\tilde F
\]

Purpose:

Test valuable + unsupported behaviour.


### Return + Delta-F

Select based on:

\[
\tilde R+\widetilde{\Delta F}
\]

Purpose:

Test valuable behaviour whose support is deteriorating rapidly.


### RAER

Use the best evidence-supported criterion discovered in Stage I.


---

# 15. First Intervention Screening

Initially run approximately 100k steps per candidate.

Purpose:

- eliminate clearly weak selection mechanisms;
- identify promising RAER variants;
- avoid spending compute on 1M-step experiments too early.

Keep:

- same SAC hyperparameters
- same learning starts
- same VEM size
- same repetition budget
- same repetition frequency
- same evaluation protocol


---

# 16. Longer Experiments

Only after the mechanism is validated:

Increase to:

- 500k steps
- eventually 1M steps where appropriate

Use multiple seeds.

Do NOT use the current 21k experiment as final performance evidence.


---

# 17. Final Comparison

Likely final methods:

- SAC
- IER / existing episode repetition baseline
- strongest relevant repetition baseline
- RAER
- possibly retention-aware replay variant if developed

Potential additional method:

## RPER

Retention-Aware Prioritized Experience Replay

Use retention information to influence replay sampling rather than active
episode repetition.

RPER should be treated as a separate mechanism from RAER.


---

# 18. Final Evaluation

Report:

- learning curves
- final performance
- late-training performance
- sample efficiency / AUC
- time-to-threshold where meaningful
- mean across seeds
- 95% confidence intervals

For multi-task experiments consider:

- IQM
- probability of improvement
- performance profiles


## Retention analysis

Report:

- F vs G
- F vs future G
- Delta-F vs future G
- F vs episode age
- entropy vs F
- entropy vs capability loss
- capability-loss frequency
- retention trajectories for selected episodes


## Efficiency

Report:

- environment interactions
- wall-clock overhead
- revisit overhead
- VEM memory overhead
- repetition frequency


---

# 19. Planned Environment Expansion

Development begins with:

`HalfCheetah-v5`

After the method works, expand to additional MuJoCo tasks.

Potential environments:

- Ant
- Humanoid
- Hopper
- Walker2d
- Swimmer

Do not expand environments before validating the mechanism on the development
environment.


---

# 20. Reproducibility Rules

Before every major experiment:

```bash
./scripts/run_tests.sh
