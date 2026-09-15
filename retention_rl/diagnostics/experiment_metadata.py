from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone

import gymnasium
import mujoco
import numpy
import stable_baselines3
import torch


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def _git_is_dirty():
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
        ).strip()
        return bool(output)
    except Exception:
        return None


def save_experiment_metadata(
    output_dir,
    env_id,
    seed,
    total_timesteps,
    model,
):
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "experiment": {
            "environment": env_id,
            "seed": seed,
            "total_timesteps": total_timesteps,
            "started_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },

        "algorithm": {
            "name": "SAC",
            "learning_rate": float(model.learning_rate),
            "buffer_size": int(model.buffer_size),
            "learning_starts": int(model.learning_starts),
            "batch_size": int(model.batch_size),
            "tau": float(model.tau),
            "gamma": float(model.gamma),
            "train_freq": str(model.train_freq),
            "gradient_steps": int(model.gradient_steps),
            "target_entropy": float(model.target_entropy),
        },

        "network": {
            "actor": str(model.actor),
            "critic": str(model.critic),
            "critic_target": str(model.critic_target),
        },

        "software": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
        },

        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_device": str(model.device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },

        "reproducibility": {
            "git_commit": _git_commit(),
            "git_dirty": _git_is_dirty(),
        },
    }

    path = os.path.join(
        output_dir,
        "metadata.json",
    )

    with open(path, "w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    return path
