"""Run a random policy on a Gymnasium environment and report its returns.

This is the baseline every trained agent must beat.

Usage:
    python agents/random_agent.py --env Pendulum-v1 --episodes 5
    python agents/random_agent.py --env Pendulum-v1 --episodes 1 --render
"""

import argparse

import gymnasium as gym
import numpy as np


def describe_env(env: gym.Env) -> None:
    print(f"Environment:       {env.spec.id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Max episode steps: {env.spec.max_episode_steps}")
    print()


def run_episode(env: gym.Env, seed: int, show_steps: int) -> tuple[float, int]:
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    episode_return = 0.0
    steps = 0

    while True:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        if steps < show_steps:
            print(
                f"  step {steps:3d} | obs={np.round(obs, 3)} "
                f"| action={np.round(action, 3)} | reward={reward:8.3f}"
            )

        episode_return += float(reward)
        steps += 1
        obs = next_obs

        if terminated or truncated:
            return episode_return, steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-steps", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="human" if args.render else None)
    describe_env(env)

    returns = []
    for ep in range(args.episodes):
        print(f"Episode {ep}")
        ep_return, ep_len = run_episode(env, seed=args.seed + ep, show_steps=args.show_steps)
        returns.append(ep_return)
        print(f"  -> return={ep_return:.1f}, length={ep_len}\n")

    env.close()
    print(f"Random policy: mean return = {np.mean(returns):.1f} +/- {np.std(returns):.1f} "
          f"over {args.episodes} episodes")


if __name__ == "__main__":
    main()
