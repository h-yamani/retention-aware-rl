from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from retention_rl.retention.metrics import (
    retention_deficit,
)
from retention_rl.retention.stored_action_log_prob import (
    trajectory_nll,
)


class RetentionTracker:
    """
    Measure Stage-I retention and capability statistics.

    For each episode currently in VEM:

        F = max(0, current_nll - insert_nll)

    When a CapabilityEvaluator is supplied:

        G = stored_return - mean_revisit_return

    Stage I is observational only.
    """

    FIELDNAMES = [
        "checkpoint_step",
        "episode_id",
        "insert_step",
        "episode_age_steps",
        "episode_return",
        "episode_length",
        "insert_nll",
        "current_nll",
        "retention_deficit",
        "revisit_mean_return",
        "revisit_std_return",
        "capability_gap",
        "vem_rank",
        "vem_size",
    ]

    def __init__(
        self,
        output_csv=None,
    ):
        self.output_csv = (
            Path(output_csv)
            if output_csv is not None
            else None
        )

        self.history = []

        if self.output_csv is not None:
            self.output_csv.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

    def _append_csv(
        self,
        rows,
    ):
        if self.output_csv is None:
            return

        if not rows:
            return

        write_header = (
            not self.output_csv.exists()
            or self.output_csv.stat().st_size == 0
        )

        with self.output_csv.open(
            "a",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.FIELDNAMES,
            )

            if write_header:
                writer.writeheader()

            writer.writerows(
                rows
            )

    def evaluate(
        self,
        actor,
        vem,
        checkpoint_step: int,
        capability_evaluator=None,
        model=None,
    ):
        """
        Evaluate every episode currently retained in VEM.

        capability_evaluator/model are optional so the existing
        retention-only unit tests remain valid.
        """

        if (
            capability_evaluator is not None
            and model is None
        ):
            raise ValueError(
                "model must be provided when "
                "capability_evaluator is used."
            )

        rows = []

        episodes = list(
            vem.episodes
        )

        vem_size = len(
            episodes
        )

        for rank, episode in enumerate(
            episodes,
            start=1,
        ):
            states = np.asarray(
                episode.states,
                dtype=np.float32,
            )

            actions = np.asarray(
                episode.actions,
                dtype=np.float32,
            )

            if len(states) != len(actions) + 1:
                raise RuntimeError(
                    "Invalid stored episode: expected "
                    "len(states) == len(actions) + 1."
                )

            current_nll = trajectory_nll(
                actor=actor,
                states=states[:-1],
                actions=actions,
            )

            deficit = retention_deficit(
                insert_nll=episode.insert_nll,
                current_nll=current_nll,
            )

            episode.last_nll = float(
                current_nll
            )

            episode.last_retention_deficit = float(
                deficit
            )

            revisit_mean_return = np.nan
            revisit_std_return = np.nan
            gap = np.nan

            if capability_evaluator is not None:
                capability_result = (
                    capability_evaluator.evaluate(
                        model=model,
                        episode=episode,
                    )
                )

                revisit_mean_return = float(
                    capability_result.revisit_mean_return
                )

                revisit_std_return = float(
                    capability_result.revisit_std_return
                )

                gap = float(
                    capability_result.capability_gap
                )

            row = {
                "checkpoint_step": int(
                    checkpoint_step
                ),
                "episode_id": int(
                    episode.episode_id
                ),
                "insert_step": int(
                    episode.insert_step
                ),
                "episode_age_steps": int(
                    checkpoint_step
                    - episode.insert_step
                ),
                "episode_return": float(
                    episode.episode_return
                ),
                "episode_length": int(
                    len(episode.actions)
                ),
                "insert_nll": float(
                    episode.insert_nll
                ),
                "current_nll": float(
                    current_nll
                ),
                "retention_deficit": float(
                    deficit
                ),
                "revisit_mean_return": (
                    revisit_mean_return
                ),
                "revisit_std_return": (
                    revisit_std_return
                ),
                "capability_gap": gap,
                "vem_rank": int(
                    rank
                ),
                "vem_size": int(
                    vem_size
                ),
            }

            rows.append(
                row
            )

            self.history.append(
                row
            )

        self._append_csv(
            rows
        )

        return rows
