import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from retention_rl.diagnostics.detailed_evaluation import (
    DetailedEvaluationCallback,
)
from retention_rl.diagnostics.experiment_metadata import (
    save_experiment_metadata,
)
from retention_rl.diagnostics.sac_diagnostics import (
    SACDiagnosticsCallback,
)


def make_env(
    env_id,
    seed,
    monitor_file=None,
):
    env = gym.make(
        env_id,
    )

    env = Monitor(
        env,
        filename=monitor_file,
        info_keywords=(),
    )

    env.reset(
        seed=seed,
    )

    env.action_space.seed(
        seed,
    )

    return env


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        default="HalfCheetah-v5",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--diagnostic-freq",
        type=int,
        default=1_000,
    )

    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
    )

    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Run directories
    # ---------------------------------------------------------

    run_name = (
        f"{args.env}"
        f"_steps{args.steps}"
        f"_seed{args.seed}"
    )

    result_dir = os.path.join(
        "results",
        "baseline_sac",
        run_name,
    )

    model_dir = os.path.join(
        result_dir,
        "models",
    )

    checkpoint_dir = os.path.join(
        result_dir,
        "checkpoints",
    )

    eval_dir = os.path.join(
        result_dir,
        "eval",
    )

    log_dir = os.path.join(
        result_dir,
        "logs",
    )

    diagnostic_dir = os.path.join(
        result_dir,
        "diagnostics",
    )

    metadata_dir = os.path.join(
        result_dir,
        "metadata",
    )

    monitor_dir = os.path.join(
        result_dir,
        "monitor",
    )

    tensorboard_dir = os.path.join(
        result_dir,
        "tensorboard",
    )

    directories = [
        model_dir,
        checkpoint_dir,
        eval_dir,
        log_dir,
        diagnostic_dir,
        metadata_dir,
        monitor_dir,
        tensorboard_dir,
    ]

    for directory in directories:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    # ---------------------------------------------------------
    # Training and standard evaluation environments
    # ---------------------------------------------------------

    train_monitor = os.path.join(
        monitor_dir,
        "train",
    )

    eval_monitor = os.path.join(
        monitor_dir,
        "eval",
    )

    train_env = make_env(
        args.env,
        args.seed,
        train_monitor,
    )

    eval_env = make_env(
        args.env,
        args.seed + 10_000,
        eval_monitor,
    )

    # ---------------------------------------------------------
    # SAC baseline
    # ---------------------------------------------------------

    model = SAC(
        policy="MlpPolicy",
        env=train_env,

        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=5_000,
        batch_size=256,

        tau=0.005,
        gamma=0.99,

        train_freq=1,
        gradient_steps=1,

        ent_coef="auto",
        target_entropy="auto",

        policy_kwargs=dict(
            net_arch=[
                256,
                256,
            ],
        ),

        seed=args.seed,
        device="auto",

        tensorboard_log=tensorboard_dir,

        verbose=1,
    )

    # ---------------------------------------------------------
    # Save t=0 model
    # ---------------------------------------------------------

    initial_model_path = os.path.join(
        checkpoint_dir,
        "initial_model",
    )

    model.save(
        initial_model_path,
    )

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------

    logger = configure(
        log_dir,
        [
            "stdout",
            "csv",
            "tensorboard",
        ],
    )

    model.set_logger(
        logger,
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata_path = save_experiment_metadata(
        output_dir=metadata_dir,
        env_id=args.env,
        seed=args.seed,
        total_timesteps=args.steps,
        model=model,
    )

    # ---------------------------------------------------------
    # Standard SB3 evaluation
    #
    # Preserves raw returns in evaluations.npz and saves
    # best_model.zip.
    # ---------------------------------------------------------

    eval_callback = EvalCallback(
        eval_env,

        best_model_save_path=model_dir,
        log_path=eval_dir,

        eval_freq=args.eval_freq,

        n_eval_episodes=args.eval_episodes,

        deterministic=True,
        render=False,
    )

    # ---------------------------------------------------------
    # Detailed behavioral evaluation
    #
    # Uses an independent environment and does not affect
    # training experience.
    # ---------------------------------------------------------

    detailed_eval_callback = (
        DetailedEvaluationCallback(
            env_id=args.env,
            output_dir=eval_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            seed=args.seed + 20_000,
            verbose=1,
        )
    )

    # ---------------------------------------------------------
    # Model checkpoints
    # ---------------------------------------------------------

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,

        save_path=checkpoint_dir,

        name_prefix="sac",

        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # ---------------------------------------------------------
    # SAC diagnostics
    # ---------------------------------------------------------

    diagnostic_callback = (
        SACDiagnosticsCallback(
            output_dir=diagnostic_dir,
            diagnostic_freq=args.diagnostic_freq,
            verbose=1,
        )
    )

    callbacks = CallbackList(
        [
            eval_callback,
            detailed_eval_callback,
            checkpoint_callback,
            diagnostic_callback,
        ]
    )

    # ---------------------------------------------------------
    # Experiment summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RESEARCH BASELINE — SAC")
    print("=" * 70)

    print(
        "Environment:",
        args.env,
    )

    print(
        "Steps:",
        args.steps,
    )

    print(
        "Seed:",
        args.seed,
    )

    print()

    print(
        "Evaluation frequency:",
        args.eval_freq,
    )

    print(
        "Evaluation episodes:",
        args.eval_episodes,
    )

    print(
        "Diagnostic frequency:",
        args.diagnostic_freq,
    )

    print(
        "Checkpoint frequency:",
        args.checkpoint_freq,
    )

    print()

    print(
        "Device:",
        model.device,
    )

    print(
        "Results:",
        result_dir,
    )

    print(
        "Metadata:",
        metadata_path,
    )

    print(
        "Initial model:",
        initial_model_path + ".zip",
    )

    print("=" * 70)
    print()

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
    )

    # ---------------------------------------------------------
    # Save final model
    # ---------------------------------------------------------

    final_path = os.path.join(
        model_dir,
        "final_model",
    )

    model.save(
        final_path,
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Results:",
        result_dir,
    )

    print(
        "Initial model:",
        initial_model_path + ".zip",
    )

    print(
        "Final model:",
        final_path + ".zip",
    )

    print(
        "Training log:",
        os.path.join(
            log_dir,
            "progress.csv",
        ),
    )

    print(
        "SAC diagnostics:",
        os.path.join(
            diagnostic_dir,
            "sac_diagnostics.csv",
        ),
    )

    print(
        "Raw SB3 evaluation:",
        os.path.join(
            eval_dir,
            "evaluations.npz",
        ),
    )

    print(
        "Detailed evaluation:",
        os.path.join(
            eval_dir,
            "detailed_evaluation.csv",
        ),
    )

    print(
        "Evaluation summary:",
        os.path.join(
            eval_dir,
            "evaluation_summary.csv",
        ),
    )

    print(
        "Metadata:",
        metadata_path,
    )

    print("=" * 70)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
