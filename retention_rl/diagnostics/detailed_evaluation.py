from __future__ import annotations

import csv
import os

import gymnasium as gym
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class DetailedEvaluationCallback(BaseCallback):
    """
    Independent deterministic evaluation for research analysis.

    This callback does not modify:
      - the training environment
      - the replay buffer
      - SAC losses
      - optimizers
      - policy updates

    For each evaluation episode it records:
      - episodic return
      - episode length
      - forward reward
      - control reward/cost
      - mean x velocity
      - action magnitude
      - action saturation
    """

    def __init__(
        self,
        env_id: str,
        output_dir: str,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        seed: int = 20000,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.env_id = env_id
        self.output_dir = output_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed

        self.csv_path = os.path.join(
            output_dir,
            "detailed_evaluation.csv",
        )

        self.summary_csv_path = os.path.join(
            output_dir,
            "evaluation_summary.csv",
        )

        self._last_eval_step = -1

        self.episode_fields = [
            "timesteps",
            "episode",
            "episode_seed",
            "return",
            "episode_length",
            "forward_reward",
            "control_reward",
            "mean_x_velocity",
            "action_abs_mean",
            "action_saturation_fraction",
        ]

        self.summary_fields = [
            "timesteps",
            "n_episodes",
            "return_mean",
            "return_std",
            "return_median",
            "return_min",
            "return_max",
            "return_q25",
            "return_q75",
            "episode_length_mean",
            "forward_reward_mean",
            "control_reward_mean",
            "mean_x_velocity",
            "action_abs_mean",
            "action_saturation_fraction_mean",
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
                    fieldnames=self.episode_fields,
                )
                writer.writeheader()

        if not os.path.exists(self.summary_csv_path):
            with open(
                self.summary_csv_path,
                "w",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self.summary_fields,
                )
                writer.writeheader()

    def _evaluate_episode(
        self,
        episode_index: int,
    ):
        env = gym.make(self.env_id)

        episode_seed = (
            self.seed
            + self.num_timesteps
            + episode_index
        )

        obs, _ = env.reset(
            seed=episode_seed,
        )

        env.action_space.seed(
            episode_seed,
        )

        done = False

        episode_return = 0.0
        episode_length = 0

        forward_reward_total = 0.0
        control_reward_total = 0.0
        x_velocity_total = 0.0

        action_abs_sum = 0.0
        action_count = 0
        saturated_action_count = 0

        while not done:
            action, _ = self.model.predict(
                obs,
                deterministic=True,
            )

            action_array = np.asarray(
                action,
                dtype=np.float64,
            )

            action_abs = np.abs(
                action_array,
            )

            action_abs_sum += float(
                action_abs.sum()
            )

            action_count += int(
                action_abs.size
            )

            saturated_action_count += int(
                np.sum(
                    action_abs > 0.95
                )
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            done = (
                terminated
                or truncated
            )

            episode_return += float(
                reward
            )

            episode_length += 1

            forward_reward_total += float(
                info.get(
                    "reward_forward",
                    0.0,
                )
            )

            control_reward_total += float(
                info.get(
                    "reward_ctrl",
                    0.0,
                )
            )

            x_velocity_total += float(
                info.get(
                    "x_velocity",
                    0.0,
                )
            )

        env.close()

        if action_count > 0:
            action_abs_mean = (
                action_abs_sum
                / action_count
            )

            action_saturation_fraction = (
                saturated_action_count
                / action_count
            )
        else:
            action_abs_mean = 0.0
            action_saturation_fraction = 0.0

        if episode_length > 0:
            mean_x_velocity = (
                x_velocity_total
                / episode_length
            )
        else:
            mean_x_velocity = 0.0

        return {
            "timesteps": int(
                self.num_timesteps
            ),

            "episode": int(
                episode_index
            ),

            "episode_seed": int(
                episode_seed
            ),

            "return": float(
                episode_return
            ),

            "episode_length": int(
                episode_length
            ),

            "forward_reward": float(
                forward_reward_total
            ),

            "control_reward": float(
                control_reward_total
            ),

            "mean_x_velocity": float(
                mean_x_velocity
            ),

            "action_abs_mean": float(
                action_abs_mean
            ),

            "action_saturation_fraction": float(
                action_saturation_fraction
            ),
        }

    def _run_evaluation(self):
        rows = []

        for episode_index in range(
            self.n_eval_episodes
        ):
            rows.append(
                self._evaluate_episode(
                    episode_index
                )
            )

        with open(
            self.csv_path,
            "a",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.episode_fields,
            )

            writer.writerows(rows)

        returns = np.asarray(
            [
                row["return"]
                for row in rows
            ],
            dtype=np.float64,
        )

        lengths = np.asarray(
            [
                row["episode_length"]
                for row in rows
            ],
            dtype=np.float64,
        )

        forward_rewards = np.asarray(
            [
                row["forward_reward"]
                for row in rows
            ],
            dtype=np.float64,
        )

        control_rewards = np.asarray(
            [
                row["control_reward"]
                for row in rows
            ],
            dtype=np.float64,
        )

        velocities = np.asarray(
            [
                row["mean_x_velocity"]
                for row in rows
            ],
            dtype=np.float64,
        )

        action_abs = np.asarray(
            [
                row["action_abs_mean"]
                for row in rows
            ],
            dtype=np.float64,
        )

        saturation = np.asarray(
            [
                row[
                    "action_saturation_fraction"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        summary = {
            "timesteps": int(
                self.num_timesteps
            ),

            "n_episodes": int(
                self.n_eval_episodes
            ),

            "return_mean": float(
                returns.mean()
            ),

            "return_std": float(
                returns.std()
            ),

            "return_median": float(
                np.median(returns)
            ),

            "return_min": float(
                returns.min()
            ),

            "return_max": float(
                returns.max()
            ),

            "return_q25": float(
                np.quantile(
                    returns,
                    0.25,
                )
            ),

            "return_q75": float(
                np.quantile(
                    returns,
                    0.75,
                )
            ),

            "episode_length_mean": float(
                lengths.mean()
            ),

            "forward_reward_mean": float(
                forward_rewards.mean()
            ),

            "control_reward_mean": float(
                control_rewards.mean()
            ),

            "mean_x_velocity": float(
                velocities.mean()
            ),

            "action_abs_mean": float(
                action_abs.mean()
            ),

            "action_saturation_fraction_mean":
                float(
                    saturation.mean()
                ),
        }

        with open(
            self.summary_csv_path,
            "a",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.summary_fields,
            )

            writer.writerow(summary)

        if self.verbose:
            print(
                "[Detailed Eval] "
                f"step={self.num_timesteps} "
                f"return={summary['return_mean']:.2f} "
                f"velocity={summary['mean_x_velocity']:.3f} "
                f"forward={summary['forward_reward_mean']:.2f} "
                f"ctrl={summary['control_reward_mean']:.2f}"
            )

    def _on_step(self) -> bool:
        if (
            self.num_timesteps
            % self.eval_freq
            != 0
        ):
            return True

        if (
            self.num_timesteps
            == self._last_eval_step
        ):
            return True

        self._run_evaluation()

        self._last_eval_step = (
            self.num_timesteps
        )

        return True
