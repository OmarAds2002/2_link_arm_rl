"""
env/arm_env.py

Gymnasium environment for a planar 2-link robotic arm simulated in MuJoCo.

Observation space (8D):
  [q1, q2, dq1, dq2, ee_x, ee_y, target_x, target_y]

Action space (2D):
  Normalised joint torques in [-1, 1] for shoulder and elbow.
  Scaled by gear=3 in the MuJoCo XML actuator definition.

Reward shaping:
  - Dense distance penalty     : -dist          (primary signal)
  - Progress bonus             : +0.5 * dist   (reward closing in)
  - Action smoothness penalty  : -0.005 * ||a||²
  - Settling penalty (near tgt): -0.02 * ||dq||² when dist < 0.15 m
  - Success bonus              : +20.0 on staying within threshold for 3 steps

Episode termination:
  - Terminated : EE held within success_threshold for 3 consecutive steps.
  - Truncated  : max_steps reached without success.
"""

import mujoco
import gymnasium as gym
import numpy as np


class TwoLinkArmEnv(gym.Env):
    """
    Planar 2-link arm environment.

    The arm has two hinge joints (shoulder, elbow). The goal is to move the
    end-effector (EE) to a randomised target position in the reachable workspace.

    Args:
        randomize_target (bool): If True, a new target is sampled each episode
            from a uniform distribution over the reachable workspace
            (radius 0.25 – 0.85 m). If False, the fixed target [0.8, 0.3] is
            used — useful for deterministic evaluation.
    """

    # ── Arm geometry (must match twolink.xml) ─────────────────────────────────
    L1 = 0.5   # link-1 length (m)
    L2 = 0.5   # link-2 length (m)
    MAX_REACH = L1 + L2   # 1.0 m — absolute workspace boundary

    # ── Episode parameters ────────────────────────────────────────────────────
    MAX_STEPS         = 200    # hard truncation limit per episode
    SUCCESS_THRESHOLD = 0.05   # metres — EE must be within this to count
    SUCCESS_HOLD      = 3      # consecutive steps inside threshold → terminated
    MIN_START_DIST    = 0.25   # metres — EE start must be this far from target

    # ── Reward coefficients ───────────────────────────────────────────────────
    PROGRESS_COEF    = 0.5    # weight on Δdist progress bonus
    ACTION_PENALTY   = 0.005  # weight on action magnitude penalty
    VELOCITY_PENALTY = 0.02   # weight on joint velocity penalty (near target)
    VELOCITY_ZONE    = 0.15   # metres — velocity penalty activates within this
    SUCCESS_BONUS    = 20.0   # one-time reward on successful hold

    def __init__(self, randomize_target: bool = True):
        self.model = mujoco.MjModel.from_xml_path("env/twolink.xml")
        self.data  = mujoco.MjData(self.model)

        # Cache site IDs — resolved once at construction to avoid per-step lookups
        self.ee_site_id     = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")

        if self.ee_site_id < 0:
            raise RuntimeError("Site 'end_effector' not found in twolink.xml")
        if self.target_site_id < 0:
            raise RuntimeError("Site 'target' not found in twolink.xml")

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)
        self.action_space      = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        self.randomize_target = randomize_target

        # Episode state — also fully reset in reset()
        self.target          = np.array([0.8, 0.3, 0.0], dtype=np.float64)
        self.current_step    = 0
        self.success_counter = 0
        self.prev_dist       = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_ee_pos(self) -> np.ndarray:
        """Return the current end-effector position in world frame (x, y, z)."""
        return self.data.site_xpos[self.ee_site_id].copy()

    def _sync_target_site(self):
        """
        Move the visual target site to self.target.

        MuJoCo's data.site_xpos is a read-only output computed by mj_forward —
        it cannot be written directly. The authoritative position is stored in
        model.site_pos, which mj_forward then propagates into data.site_xpos
        and the renderer.
        """
        self.model.site_pos[self.target_site_id] = self.target
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self) -> np.ndarray:
        """
        Build the 8-dimensional observation vector.

        Returns:
            np.ndarray: [q1, q2, dq1, dq2, ee_x, ee_y, target_x, target_y]
        """
        ee_pos = self._get_ee_pos()
        return np.array([
            self.data.qpos[0],   # shoulder angle (rad)
            self.data.qpos[1],   # elbow angle    (rad)
            self.data.qvel[0],   # shoulder velocity (rad/s)
            self.data.qvel[1],   # elbow velocity    (rad/s)
            ee_pos[0],           # EE x (m)
            ee_pos[1],           # EE y (m)
            self.target[0],      # target x (m)
            self.target[1],      # target y (m)
        ], dtype=np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def step(self, action):
        """
        Advance the simulation by one timestep.

        Args:
            action (np.ndarray): Joint torques in [-1, 1] for [shoulder, elbow].

        Returns:
            obs        : Next observation (8D).
            reward     : Shaped scalar reward.
            terminated : True if the EE held near the target for SUCCESS_HOLD steps.
            truncated  : True if MAX_STEPS was reached.
            info       : Dict with 'dist' (current EE-target distance in metres).
        """
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        ee_pos = self._get_ee_pos()
        dist   = np.linalg.norm(ee_pos[:2] - self.target[:2])

        # ── Reward shaping ────────────────────────────────────────────────────

        # Primary: dense distance penalty — always pull toward target
        reward = -dist

        # Progress bonus: reward the agent for closing the distance since last step
        if self.prev_dist is not None:
            reward += self.PROGRESS_COEF * (self.prev_dist - dist)
        self.prev_dist = dist

        # Smoothness: penalise large torques to encourage efficient motion
        reward -= self.ACTION_PENALTY * np.sum(np.square(action))

        # Settling: penalise high joint velocities when near the target to
        # discourage oscillation around the goal
        if dist < self.VELOCITY_ZONE:
            reward -= self.VELOCITY_PENALTY * np.sum(np.square(self.data.qvel))

        # ── Termination ───────────────────────────────────────────────────────

        terminated = False

        if dist < self.SUCCESS_THRESHOLD:
            self.success_counter += 1
        else:
            self.success_counter = 0   # reset hold counter if EE leaves zone

        if self.success_counter >= self.SUCCESS_HOLD:
            reward    += self.SUCCESS_BONUS
            terminated = True

        truncated = self.current_step >= self.MAX_STEPS

        return self._get_obs(), float(reward), terminated, truncated, {"dist": float(dist)}

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Samples a fresh target position and a starting joint configuration that
        places the EE at least MIN_START_DIST metres from the target, preventing
        trivially short episodes.

        Args:
            seed    : RNG seed forwarded to the parent Gymnasium class.
            options : Unused; kept for API compatibility.

        Returns:
            obs  : Initial observation (8D).
            info : Empty dict.
        """
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step    = 0
        self.prev_dist       = None
        self.success_counter = 0

        # ── Sample target ─────────────────────────────────────────────────────
        if self.randomize_target:
            angle = self.np_random.uniform(0.0, 2 * np.pi)
            r     = self.np_random.uniform(0.25, 0.85)
            self.target = np.array(
                [r * np.cos(angle), r * np.sin(angle), 0.0], dtype=np.float64
            )
        else:
            # Fixed target used during deterministic evaluation
            self.target = np.array([0.8, 0.3, 0.0], dtype=np.float64)

        self._sync_target_site()

        # ── Sample starting configuration ─────────────────────────────────────
        # Reject configurations where the EE is too close to the target so the
        # policy always has a non-trivial reaching task from the first step.
        for _ in range(500):
            self.data.qpos[:] = self.np_random.uniform(
                -np.pi / 2, np.pi / 2, size=self.model.nq
            )
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            if np.linalg.norm(self._get_ee_pos()[:2] - self.target[:2]) >= self.MIN_START_DIST:
                break

        self.prev_dist = np.linalg.norm(self._get_ee_pos()[:2] - self.target[:2])
        return self._get_obs(), {}