# rl_training/evaluate.py
"""
Evaluation script for a 2-link robotic arm using a trained PPO policy
with Tianshou. This script loads a saved checkpoint, constructs the same
actor and critic networks as used in training, and evaluates the policy
on fixed-target environments. It logs per-step distance and reward, and
summarizes overall performance across multiple episodes.

"""

import torch
import numpy as np
import time

from tianshou.algorithm import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Batch
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo

from env.arm_env import TwoLinkArmEnv

# Path to the trained PPO checkpoint
CHECKPOINT = "checkpoint/ppo_arm.pth"


def evaluate(num_episodes: int = 5, render: bool = False) -> None:
    """
    Evaluate the trained PPO policy on the 2-link arm environment.

    Args:
        num_episodes (int): Number of evaluation episodes
        render (bool): Whether to render the simulation with a small delay
    """
    # Detect device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 1. Environment setup =====
    # Fixed target for consistent evaluation
    env = TwoLinkArmEnv(randomize_target=False)

    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape

    # ===== 2. Actor & Critic network (must match train.py) =====
    net_a = Net(state_shape=state_shape, hidden_sizes=[128, 128])
    actor = ContinuousActorProbabilistic(
        preprocess_net=net_a,
        action_shape=action_shape,
        unbounded=True,
    ).to(device)

    net_c = Net(state_shape=state_shape, hidden_sizes=[128, 128])
    critic = ContinuousCritic(preprocess_net=net_c).to(device)

    # ===== 3. Probabilistic PPO policy =====
    policy = ProbabilisticActorPolicy(
        actor=actor,
        action_space=env.action_space,
        dist_fn=lambda loc_scale: torch.distributions.Independent(
            torch.distributions.Normal(loc=loc_scale[0], scale=loc_scale[1]), 1
        ),
        deterministic_eval=True,
        action_scaling=False,         # already scaled during training
        action_bound_method=None,     # no additional clipping
    )

    # ===== 4. PPO algorithm wrapper =====
    algorithm = PPO(
        policy=policy,
        critic=critic,
        optim=AdamOptimizerFactory(lr=3e-4),
    )

    # Load trained weights
    algorithm.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    algorithm.eval()
    print(f"✅ Loaded {CHECKPOINT}\n")

    # ===== 5. Evaluation loop =====
    ep_rewards = []  # total rewards per episode
    successes = 0    # count of episodes that reached the target

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        step = 0

        while True:
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                batch_out = policy(Batch(obs=obs_tensor, info={}))

            action = batch_out.act[0].cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            step += 1

            # Log step distance & reward
            dist = info["dist"]
            print(f"  Ep {ep+1} | Step {step:3d} | dist: {dist:.4f} | reward: {reward:.3f}")

            # Render simulation with optional delay
            if render:
                time.sleep(0.02)

            # Termination conditions
            if terminated or truncated:
                status = "✅ REACHED" if terminated else "⏱ TIMEOUT"
                print(f"{status} — Episode {ep+1} total reward: {ep_reward:.2f}\n")
                ep_rewards.append(ep_reward)
                if terminated:
                    successes += 1
                break

    # ===== 6. Evaluation summary =====
    print("=" * 40)
    print(f"Episodes:      {num_episodes}")
    print(f"Success rate:  {successes}/{num_episodes}")
    print(f"Avg reward:    {np.mean(ep_rewards):.2f}")
    print(f"Std reward:    {np.std(ep_rewards):.2f}")
    print(f"Best episode:  {np.max(ep_rewards):.2f}")
    print(f"Worst episode: {np.min(ep_rewards):.2f}")


if __name__ == "__main__":
    # Evaluate 5 episodes by default, no rendering
    evaluate(num_episodes=5, render=False)