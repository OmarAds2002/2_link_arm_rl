# rl_training/train.py
"""
Training script for a 2-link robotic arm using PPO (Proximal Policy Optimization)
with Tianshou. This script sets up vectorized environments, constructs actor and
critic networks, defines the PPO policy, and performs multi-phase training with
learning rate scheduling. Training progress is logged using TensorBoard.
"""

import os
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo

from env.arm_env import TwoLinkArmEnv

# Path to save the best trained model checkpoint
CHECKPOINT = "checkpoint/ppo_arm.pth"


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """
    Set the learning rate for all parameter groups in an optimizer.

    Args:
        optimizer (torch.optim.Optimizer): PyTorch optimizer instance
        lr (float): Learning rate to set
    """
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def train() -> None:
    """
    Main training function for the 2-link arm.

    Steps:
    1. Detect device (CUDA/CPU)
    2. Initialize vectorized training and testing environments
    3. Construct actor and critic neural networks
    4. Define the probabilistic PPO policy
    5. Setup PPO algorithm with optimizer and hyperparameters
    6. Configure collectors for data collection
    7. Execute 3-phase learning rate schedule training
    8. Save best-performing model checkpoints
    9. Log metrics to TensorBoard
    """
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Vectorized environments for training and testing
    num_train_envs = 16
    num_test_envs = 10

    train_envs = DummyVectorEnv([
        lambda: TwoLinkArmEnv(randomize_target=True) for _ in range(num_train_envs)
    ])
    test_envs = DummyVectorEnv([
        lambda: TwoLinkArmEnv(randomize_target=False) for _ in range(num_test_envs)
    ])

    # Temporary environment to extract space information
    tmp_env = TwoLinkArmEnv()
    space_info = SpaceInfo.from_env(tmp_env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape

    # ----- Actor network -----
    net_a = Net(state_shape=state_shape, hidden_sizes=[128, 128])
    actor = ContinuousActorProbabilistic(
        preprocess_net=net_a,
        action_shape=action_shape,
        unbounded=True,  # actions can go beyond [-1,1] before scaling
    ).to(device)

    # ----- Critic network -----
    net_c = Net(state_shape=state_shape, hidden_sizes=[128, 128])
    critic = ContinuousCritic(preprocess_net=net_c).to(device)

    # ----- PPO policy setup -----
    policy = ProbabilisticActorPolicy(
        actor=actor,
        action_space=tmp_env.action_space,
        dist_fn=lambda loc_scale: torch.distributions.Independent(
            torch.distributions.Normal(loc=loc_scale[0], scale=loc_scale[1]), 1
        ),
        deterministic_eval=True,
        action_scaling=True,
        action_bound_method="clip",
    )

    # ----- PPO algorithm setup -----
    algorithm = PPO(
        policy=policy,
        critic=critic,
        optim=AdamOptimizerFactory(lr=3e-4),
        advantage_normalization=True,
        recompute_advantage=True,
        eps_clip=0.2,
        value_clip=True,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
    )

    # Access internal optimizer for learning rate schedule
    internal_optim = algorithm._optimizers[0]._optim

    # ----- Collectors -----
    train_collector = Collector[CollectStats](
        algorithm,
        train_envs,
        VectorReplayBuffer(80000, num_train_envs),
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)

    # TensorBoard logger
    logger = TensorboardLogger(SummaryWriter("log/ppo_arm"))
    best_reward = -float("inf")

    # ----- 3-phase learning rate schedule -----
    phases = [
        (500, 1e-4),   # Phase 1: exploration
        (500, 1e-5),   # Phase 2: refinement
        (0, 1e-6),     # Phase 3: fine-tuning (continue until convergence)
    ]

    for phase_idx, (phase_epochs, phase_lr) in enumerate(phases):
        set_lr(internal_optim, phase_lr)
        print(f"\n{'='*50}")
        print(f"Phase {phase_idx+1}/3 | Epochs: {phase_epochs} | LR: {phase_lr:.2e}")
        print(f"{'='*50}")

        # Run training for this phase
        result = algorithm.run_training(
            OnPolicyTrainerParams(
                training_collector=train_collector,
                test_collector=test_collector,
                max_epochs=phase_epochs,
                epoch_num_steps=10000,
                collection_step_num_env_steps=4000,
                update_step_num_repetitions=20,
                test_step_num_episodes=20,
                batch_size=256,
                logger=logger,
            )
        )

        # Save checkpoint if improved
        phase_best = result.best_reward
        if phase_best > best_reward:
            best_reward = phase_best
            os.makedirs("checkpoint", exist_ok=True)
            torch.save(algorithm.state_dict(), CHECKPOINT)
            print(f"✅ New best: {best_reward:.3f} — saved")

    print(f"\nTraining completed | Best reward: {best_reward:.3f}")


if __name__ == "__main__":
    train()