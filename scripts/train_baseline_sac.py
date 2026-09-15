import argparse
import os

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


def make_env(env_id, seed):
    env = gym.make(env_id)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v5")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_name = f"{args.env}_seed{args.seed}"

    result_dir = os.path.join(
        "results",
        "baseline_sac",
        run_name,
    )

    model_dir = os.path.join(result_dir, "models")
    eval_dir = os.path.join(result_dir, "eval")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    train_env = make_env(args.env, args.seed)
    eval_env = make_env(args.env, args.seed + 10000)

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
            net_arch=[256, 256]
        ),

        seed=args.seed,
        device="auto",
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=eval_dir,
        eval_freq=5_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    print()
    print("=" * 60)
    print("BASELINE SAC")
    print("=" * 60)
    print("Environment:", args.env)
    print("Steps:", args.steps)
    print("Seed:", args.seed)
    print("Device: auto")
    print("=" * 60)
    print()

    model.learn(
        total_timesteps=args.steps,
        callback=eval_callback,
        progress_bar=True,
    )

    final_path = os.path.join(
        model_dir,
        "final_model",
    )

    model.save(final_path)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Results:", result_dir)
    print("Final model:", final_path + ".zip")
    print("=" * 60)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
