from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def _to_tensor(
    array,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert NumPy arrays or tensors to float32 tensors
    on the SAC actor's device.
    """

    if isinstance(array, torch.Tensor):
        return array.to(
            device=device,
            dtype=torch.float32,
        )

    return torch.as_tensor(
        np.asarray(array),
        dtype=torch.float32,
        device=device,
    )


def stored_action_log_prob(
    actor,
    observations,
    actions,
) -> torch.Tensor:
    """
    Compute log pi_theta(a | s) for STORED historical actions.

    This function does NOT sample new actions.

    Parameters
    ----------
    actor:
        Stable-Baselines3 SAC actor.

    observations:
        Batch of historical observations.

        Shape:
            [batch_size, observation_dim]

    actions:
        Batch of historical actions in the actor's normalized
        squashed action space.

        For the current HalfCheetah-v5 SAC baseline this is [-1, 1].

        Shape:
            [batch_size, action_dim]

    Returns
    -------
    torch.Tensor
        Log probability for each stored state-action pair.

        Shape:
            [batch_size]

    Notes
    -----
    SB3 SAC uses a tanh-squashed diagonal Gaussian.

    The actor first predicts:

        mean(s), log_std(s)

    We then construct that distribution and evaluate the
    HISTORICAL stored action under it.

    SB3's SquashedDiagGaussianDistribution.log_prob()
    performs:

      1. inverse tanh
      2. Gaussian log probability
      3. tanh Jacobian correction

    Therefore we reuse SB3's own implementation rather than
    reimplementing the SAC probability equation.
    """

    device = actor.device

    obs_tensor = _to_tensor(
        observations,
        device,
    )

    action_tensor = _to_tensor(
        actions,
        device,
    )

    # Support a single state-action pair.
    if obs_tensor.ndim == 1:
        obs_tensor = obs_tensor.unsqueeze(0)

    if action_tensor.ndim == 1:
        action_tensor = action_tensor.unsqueeze(0)

    if obs_tensor.shape[0] != action_tensor.shape[0]:
        raise ValueError(
            "Observations and actions must have the same "
            "batch dimension."
        )

    # Stored SAC actions should be in the squashed action space.
    #
    # Numerical clipping protects inverse tanh near exactly
    # -1 and +1.
    epsilon = 1e-6

    action_tensor = torch.clamp(
        action_tensor,
        -1.0 + epsilon,
        1.0 - epsilon,
    )

    with torch.no_grad():

        (
            mean_actions,
            log_std,
            kwargs,
        ) = actor.get_action_dist_params(
            obs_tensor
        )

        distribution = (
            actor.action_dist.proba_distribution(
                mean_actions,
                log_std,
                **kwargs,
            )
        )

        log_prob = distribution.log_prob(
            action_tensor
        )

    return log_prob


def trajectory_nll(
    actor,
    states,
    actions,
) -> float:
    """
    Compute mean negative log-likelihood of a stored trajectory.

        NLL(theta, tau)
        =
        -(1/T) * sum_j log pi_theta(a_j | s_j)

    The actions are the historical actions stored in the
    trajectory. No new policy actions are sampled.
    """

    log_prob = stored_action_log_prob(
        actor=actor,
        observations=states,
        actions=actions,
    )

    nll = -log_prob.mean()

    return float(
        nll.cpu().item()
    )


def trajectory_log_prob_stats(
    actor,
    states,
    actions,
) -> Tuple[float, float, float]:
    """
    Optional diagnostic statistics for a stored trajectory.

    Returns
    -------
    mean_log_prob
    min_log_prob
    max_log_prob
    """

    log_prob = stored_action_log_prob(
        actor=actor,
        observations=states,
        actions=actions,
    )

    return (
        float(log_prob.mean().cpu().item()),
        float(log_prob.min().cpu().item()),
        float(log_prob.max().cpu().item()),
    )
