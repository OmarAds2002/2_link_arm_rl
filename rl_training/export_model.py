import torch
import torch.nn as nn

from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic
from tianshou.utils.space_info import SpaceInfo

from env.arm_env import TwoLinkArmEnv

CHECKPOINT = "checkpoint/ppo_arm.pth"
EXPORT_PATH = "checkpoint/ppo_actor.pt"


class ActorWrapper(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # ContinuousActorProbabilistic returns ((mu, sigma), state)
        result = self.actor(obs)
        # result[0] is (mu, sigma), result[0][0] is mu
        mu = result[0][0]
        return torch.clamp(mu, -1.0, 1.0)


def export():
    device = "cpu"

    env = TwoLinkArmEnv()
    space_info = SpaceInfo.from_env(env)
    state_shape  = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape

    net = Net(state_shape=state_shape, hidden_sizes=[128, 128])
    actor = ContinuousActorProbabilistic(
        preprocess_net=net,
        action_shape=action_shape,
        unbounded=True,
    ).to(device)

    ckpt = torch.load(CHECKPOINT, map_location=device)

    print("=== All checkpoint keys ===")
    for k in ckpt:
        print(" ", k)

    # Correct prefix from the debug output
    PREFIX = "policy.actor."

    actor_dict = {
        k[len(PREFIX):]: v
        for k, v in ckpt.items()
        if k.startswith(PREFIX)
    }

    print(f"\n✅ Found {len(actor_dict)} actor keys:")
    for k in actor_dict:
        print(" ", k)

    actor.load_state_dict(actor_dict, strict=True)
    actor.eval()

    wrapped = ActorWrapper(actor)
    wrapped.eval()

    dummy = torch.randn(1, state_shape[0])

    # Verify wrapper output before tracing
    with torch.no_grad():
        out = wrapped(dummy)
    print(f"\n✅ Wrapper output shape: {out.shape}  (expected [1, {action_shape[0] if hasattr(action_shape, '__len__') else action_shape}])")

    traced = torch.jit.trace(wrapped, dummy)
    torch.jit.save(traced, EXPORT_PATH)
    print(f"✅ Exported to {EXPORT_PATH}")


if __name__ == "__main__":
    export()