from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from retention_rl.memory import ValuableEpisodicMemory
from retention_rl.retention.metrics import retention_deficit
from retention_rl.retention.stored_action_log_prob import trajectory_nll


class RetentionTracker:
    """
    Measure policy retention for episodes currently stored in VEM.

    Stage I is measurement-only:
        - no replay modification
        - no action repetition
        - no policy intervention
        - no VEM-based training

    For each stored episode i:

        current_nll_i(t)
            = -(1/T) sum log pi_theta_t(a_j | s_j)

        F_i(t)
            = max(
                0,
                current_nll_i(t) - insert_nll_i
              )
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
        "vem_rank",
        "vem_size",
    ]

    def __init__(
        self,
        output_csv: str | Path | None = None,
    ) -> None:

        self.output_csv = (
            Path(output_csv)
            if output_csv is not None
            else None
        )

        self.history: list[dict] = []

        if self.output_csv is not None:
            self.output_csv.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

    def evaluate(
        self,
        actor,
        vem: ValuableEpisodicMemory,
        checkpoint_step: int,
    ) -> list[dict]:
        """
        Re-evaluate all episodes currently in VEM
        under the current SAC actor.
        """

        rows = []

        for rank, episode in enumerate(
            vem,
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
                raise ValueError(
                    f"Episode {episode.episode_id}: "
                    "expected T+1 states for T actions."
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
                    len(actions)
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
                "vem_rank": int(
                    rank
                ),
                "vem_size": int(
                    len(vem)
                ),
            }

            rows.append(row)

        self.history.extend(
            rows
        )

        if self.output_csv is not None:
            self._append_csv(
                rows
            )

        return rows

    def _append_csv(
        self,
        rows: list[dict],
    ) -> None:

        if not rows:
            return

        assert self.output_csv is not None

        write_header = (
            not self.output_csv.exists()
            or self.output_csv.stat().st_size == 0
        )

        with self.output_csv.open(
            "a",
            newline="",
            encoding="utf-8",
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
