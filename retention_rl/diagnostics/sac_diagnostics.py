from __future__ import annotations

import csv
import os
import random
from typing import Dict

import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback


class SACDiagnosticsCallback(BaseCallback):
    """
    Research diagnostics for Stable-Baselines3 SAC.

    IMPORTANT
    ---------
    Diagnostics are observational.

    Diagnostic collection temporarily uses NumPy/PyTorch random
    numbers for replay sampling and stochastic policy sampling.
    Therefore RNG states are saved before collection and restored
    afterwards.

    This prevents diagnostic logging from advancing the random
    number streams used by subsequent SAC training.

    The callback does not modify:
      - replay-buffer contents
      - SAC losses
      - optimizer steps
      - policy parameters
      - critic parameters
      - environment interaction
    """

    def __init__(
        self,
        output_dir: str,
        diagnostic_freq: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.output_dir = output_dir
        self.diagnostic_freq = diagnostic_freq

        self.csv_path = os.path.join(
            output_dir,
            "sac_diagnostics.csv",
        )

        self._last_logged_step = -1

        self.fieldnames = [
            "timesteps",
            "updates",

            # Critic
            "q1_mean",
            "q1_std",
            "q2_mean",
            "q2_std",
            "q_min_mean",
            "q_disagreement_mean",

            # Bellman target / TD error
            "target_q_mean",
            "target_q_std",
            "td_error_q1_mean",
            "td_error_q1_std",
            "td_error_q1_abs_mean",
            "td_error_q2_mean",
            "td_error_q2_std",
            "td_error_q2_abs_mean",

            # Policy distribution
            "log_prob_mean",
            "log_prob_std",
            "entropy_estimate",
            "mu_mean",
            "mu_std",
            "log_std_mean",
            "log_std_std",

            # Actions
            "action_mean",
            "action_std",
            "action_abs_mean",
            "action_saturation_fraction",

            # Replay batch
            "reward_mean",
            "reward_std",
            "done_fraction",

            # SAC temperature
            "ent_coef",

            # Gradients remaining from the most recent
            # optimizer update.
            "last_actor_grad_norm",
            "last_critic_grad_norm",
        ]

    def _on_training_start(self) -> None:
        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        if not os.path.exists(self.csv_path):
            with open(
                self.csv_path,
                "w",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self.fieldnames,
                )

                writer.writeheader()

    @staticmethod
    def _gradient_norm(
        module: torch.nn.Module,
    ) -> float:
        total_squared_norm = 0.0

        for parameter in module.parameters():
            if parameter.grad is not None:
                grad_norm = (
                    parameter.grad
                    .detach()
                    .norm(2)
                    .item()
                )

                total_squared_norm += (
                    grad_norm ** 2
                )

        return float(
            total_squared_norm ** 0.5
        )

    def _get_entropy_coefficient(
        self,
    ) -> float:
        model = self.model

        if (
            getattr(
                model,
                "log_ent_coef",
                None,
            )
            is not None
        ):
            return float(
                torch.exp(
                    model.log_ent_coef.detach()
                )
                .cpu()
                .item()
            )

        return float(
            model.ent_coef_tensor
            .detach()
            .cpu()
            .item()
        )

    @staticmethod
    def _capture_rng_state():
        """
        Capture RNG states that diagnostic collection may consume.
        """

        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
        }

        if torch.cuda.is_available():
            state["torch_cuda"] = (
                torch.cuda.get_rng_state_all()
            )

        return state

    @staticmethod
    def _restore_rng_state(
        state,
    ):
        """
        Restore RNG states after diagnostic collection.
        """

        random.setstate(
            state["python"]
        )

        np.random.set_state(
            state["numpy"]
        )

        torch.random.set_rng_state(
            state["torch_cpu"]
        )

        if (
            torch.cuda.is_available()
            and "torch_cuda" in state
        ):
            torch.cuda.set_rng_state_all(
                state["torch_cuda"]
            )

    def _collect_diagnostics(
        self,
    ) -> Dict[str, float]:
        model = self.model
        device = model.device

        replay_data = (
            model.replay_buffer.sample(
                model.batch_size,
                env=model._vec_normalize_env,
            )
        )

        observations = (
            replay_data.observations.to(
                device
            )
        )

        actions = (
            replay_data.actions.to(
                device
            )
        )

        next_observations = (
            replay_data.next_observations.to(
                device
            )
        )

        rewards = (
            replay_data.rewards.to(
                device
            )
        )

        dones = (
            replay_data.dones.to(
                device
            )
        )

        with torch.no_grad():
            current_q_values = (
                model.critic(
                    observations,
                    actions,
                )
            )

            q1 = current_q_values[0]
            q2 = current_q_values[1]

            q_min = torch.minimum(
                q1,
                q2,
            )

            q_disagreement = torch.abs(
                q1 - q2
            )

            (
                next_actions,
                next_log_prob,
            ) = model.actor.action_log_prob(
                next_observations
            )

            next_q_values = torch.cat(
                model.critic_target(
                    next_observations,
                    next_actions,
                ),
                dim=1,
            )

            next_q_values, _ = torch.min(
                next_q_values,
                dim=1,
                keepdim=True,
            )

            ent_coef = (
                self._get_entropy_coefficient()
            )

            target_q = (
                rewards
                + (1.0 - dones)
                * model.gamma
                * (
                    next_q_values
                    - ent_coef
                    * next_log_prob.reshape(
                        -1,
                        1,
                    )
                )
            )

            td_error_q1 = (
                target_q - q1
            )

            td_error_q2 = (
                target_q - q2
            )

            (
                sampled_actions,
                log_prob,
            ) = model.actor.action_log_prob(
                observations
            )

            (
                mean_actions,
                log_std,
                _,
            ) = (
                model.actor
                .get_action_dist_params(
                    observations
                )
            )

            entropy_estimate = (
                -log_prob.mean()
            )

            action_abs = torch.abs(
                sampled_actions
            )

            saturation_fraction = (
                (
                    action_abs > 0.95
                )
                .float()
                .mean()
            )

        row = {
            "timesteps":
                int(self.num_timesteps),

            "updates":
                int(model._n_updates),

            "q1_mean":
                q1.mean().item(),

            "q1_std":
                q1.std().item(),

            "q2_mean":
                q2.mean().item(),

            "q2_std":
                q2.std().item(),

            "q_min_mean":
                q_min.mean().item(),

            "q_disagreement_mean":
                q_disagreement.mean().item(),

            "target_q_mean":
                target_q.mean().item(),

            "target_q_std":
                target_q.std().item(),

            "td_error_q1_mean":
                td_error_q1.mean().item(),

            "td_error_q1_std":
                td_error_q1.std().item(),

            "td_error_q1_abs_mean":
                td_error_q1.abs().mean().item(),

            "td_error_q2_mean":
                td_error_q2.mean().item(),

            "td_error_q2_std":
                td_error_q2.std().item(),

            "td_error_q2_abs_mean":
                td_error_q2.abs().mean().item(),

            "log_prob_mean":
                log_prob.mean().item(),

            "log_prob_std":
                log_prob.std().item(),

            "entropy_estimate":
                entropy_estimate.item(),

            "mu_mean":
                mean_actions.mean().item(),

            "mu_std":
                mean_actions.std().item(),

            "log_std_mean":
                log_std.mean().item(),

            "log_std_std":
                log_std.std().item(),

            "action_mean":
                sampled_actions.mean().item(),

            "action_std":
                sampled_actions.std().item(),

            "action_abs_mean":
                action_abs.mean().item(),

            "action_saturation_fraction":
                saturation_fraction.item(),

            "reward_mean":
                rewards.mean().item(),

            "reward_std":
                rewards.std().item(),

            "done_fraction":
                dones.float().mean().item(),

            "ent_coef":
                ent_coef,

            "last_actor_grad_norm":
                self._gradient_norm(
                    model.actor
                ),

            "last_critic_grad_norm":
                self._gradient_norm(
                    model.critic
                ),
        }

        return row

    def _collect_without_rng_side_effects(
        self,
    ) -> Dict[str, float]:
        """
        Collect diagnostics while preserving training RNG states.
        """

        rng_state = (
            self._capture_rng_state()
        )

        try:
            row = (
                self._collect_diagnostics()
            )
        finally:
            self._restore_rng_state(
                rng_state
            )

        return row

    def _on_step(self) -> bool:
        if (
            self.num_timesteps
            <= self.model.learning_starts
        ):
            return True

        if (
            self.model.replay_buffer.size()
            < self.model.batch_size
        ):
            return True

        if (
            self.num_timesteps
            % self.diagnostic_freq
            != 0
        ):
            return True

        if (
            self.num_timesteps
            == self._last_logged_step
        ):
            return True

        row = (
            self._collect_without_rng_side_effects()
        )

        with open(
            self.csv_path,
            "a",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.fieldnames,
            )

            writer.writerow(
                row
            )

        self._last_logged_step = (
            self.num_timesteps
        )

        if self.verbose:
            print(
                "[Diagnostics] "
                f"step={self.num_timesteps} "
                f"Q={row['q_min_mean']:.3f} "
                f"TD={row['td_error_q1_abs_mean']:.3f} "
                f"H={row['entropy_estimate']:.3f}"
            )

        return True
