"""Retention and capability metrics used in Stage I."""

from __future__ import annotations


def retention_deficit(insert_nll: float, current_nll: float) -> float:
    """Positive increase in episode NLL since VEM insertion.

    F_t(tau) = max(0, NLL_t(tau) - NLL_store(tau))

    Larger values mean that the stored successful actions receive less support
    from the current policy than they did when the episode entered VEM.
    """
    return max(0.0, float(current_nll) - float(insert_nll))


def capability_gap(
    stored_return: float,
    revisit_mean_return: float,
) -> float:
    """Behavioural capability gap from the same initial condition.

    G_t(tau) = R(tau) - mean_revisit_return_t(tau)

    A large positive value means the current policy performs worse from the
    original initial condition than the historically valuable episode did.
    """
    return float(stored_return) - float(revisit_mean_return)
