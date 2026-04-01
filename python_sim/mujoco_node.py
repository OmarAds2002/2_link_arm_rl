"""
python_sim/mujoco_node.py

ROS2 node for simulating a 2-link robotic arm using MuJoCo. This node:
- Loads the MuJoCo model from an XML file.
- Publishes joint states and target positions.
- Subscribes to torque commands for the joints.
- Runs a high-frequency simulation loop with optional viewer rendering.
- Handles episode resets and success/timeout conditions.
"""

import os
import sys
import time
import threading
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# Ensure parent directory is in path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Simulation parameters
MIN_START_DIST = 0.25
MAX_STEPS      = 200
SUCCESS_DIST   = 0.05
SUCCESS_HOLD   = 3
RESET_DELAY    = 1.5
SIM_HZ         = 500
RENDER_HZ      = 60


class MujocoNode(Node):
    """
    ROS2 node to simulate and control a 2-link arm in MuJoCo.
    Publishes joint states, subscribes to torque commands, and optionally
    displays a viewer.
    """

    def __init__(self) -> None:
        super().__init__('mujoco_node')

        # Load MuJoCo model
        xml_path = os.path.join(os.path.dirname(__file__), '..', 'env', 'twolink.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # Get site IDs for end-effector and target
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")

        if self.ee_site_id < 0 or self.target_site_id < 0:
            raise RuntimeError("Required sites 'end_effector' or 'target' not found in XML.")

        # Thread locks for safe multi-threading
        self._ctrl_lock = threading.Lock()
        self._physics_lock = threading.Lock()

        # Control and state variables
        self.ctrl = np.zeros(self.model.nu)
        self.target: np.ndarray = np.zeros(3)
        self.current_step = 0
        self.episode_count = 0
        self.success_count = 0
        self.needs_reset = False
        self.reset_time = 0.0

        # ── ROS2 publishers & subscribers ─────────────────────────────
        self.joint_pub  = self.create_publisher(JointState, '/joint_states', 10)
        self.target_pub = self.create_publisher(Float64MultiArray, '/target_pos', 10)

        self.cmd_sub = self.create_subscription(
            Float64MultiArray, '/joint_torque_cmd', self.cmd_callback, 10
        )

        # ── Viewer ───────────────────────────────────────────────────
        self._viewer_ready = threading.Event()
        threading.Thread(target=self._run_viewer, daemon=True).start()
        self._viewer_ready.wait()

        # Initial reset
        self._reset()
        # High-frequency simulation timer
        self.create_timer(1.0 / SIM_HZ, self._sim_step)
        self.get_logger().info('MuJoCo node ready')

    # ── Reset & Target Management ──────────────────────────────────
    def _reset(self) -> None:
        """
        Reset the simulation with a new target and randomized arm configuration.
        Ensures starting distance is not too close.
        """
        rng = np.random.default_rng()

        # Random target in circle
        angle = rng.uniform(0.0, 2 * np.pi)
        r = rng.uniform(0.25, 0.85)
        target = np.array([r * np.cos(angle), r * np.sin(angle), 0.0])

        # Find valid initial configuration
        tmp = mujoco.MjData(self.model)
        dist = 0.0
        for _ in range(500):
            tmp.qpos[:] = rng.uniform(-np.pi / 2, np.pi / 2, size=self.model.nq)
            tmp.qvel[:] = 0.0
            mujoco.mj_forward(self.model, tmp)
            dist = np.linalg.norm(tmp.site_xpos[self.ee_site_id][:2] - target[:2])
            if dist >= MIN_START_DIST:
                break

        self.model.site_pos[self.target_site_id] = target

        # Apply reset safely
        with self._physics_lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = tmp.qpos
            self.data.qvel[:] = tmp.qvel
            mujoco.mj_forward(self.model, self.data)
            self.target = target
            self.current_step = 0
            self.success_count = 0
            self.episode_count += 1

        with self._ctrl_lock:
            self.ctrl[:] = 0.0

        # Publish new target to ROS
        self._publish_target(target)

        self.get_logger().info(
            f"Episode {self.episode_count} | "
            f"target: [{target[0]:.2f}, {target[1]:.2f}] | "
            f"start dist: {dist:.2f} m"
        )

    def _publish_target(self, target: np.ndarray) -> None:
        """Publish the target position as a ROS Float64MultiArray."""
        msg = Float64MultiArray()
        msg.data = [float(target[0]), float(target[1])]
        self.target_pub.publish(msg)

    # ── Viewer Thread ─────────────────────────────────────────────
    def _run_viewer(self) -> None:
        """Launch MuJoCo passive viewer in a separate thread."""
        with mujoco.viewer.launch_passive(self.model, self.data,
                                          show_left_ui=False, show_right_ui=False) as viewer:
            # Set camera view
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.5
            viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
            self._viewer_ready.set()

            while viewer.is_running():
                with self._physics_lock:
                    viewer.sync()
                time.sleep(1.0 / RENDER_HZ)

        self.get_logger().warn('Viewer closed')
        rclpy.shutdown()

    # ── ROS2 Subscriber Callback ─────────────────────────────────
    def cmd_callback(self, msg: Float64MultiArray) -> None:
        """Receive joint torque commands and store them for the simulation step."""
        with self._ctrl_lock:
            arr = np.array(msg.data, dtype=np.float64)
            n = min(len(arr), self.model.nu)
            self.ctrl[:n] = arr[:n]

    # ── Simulation Step ──────────────────────────────────────────
    def _sim_step(self) -> None:
        """Perform one simulation step, update joint states, and handle success/timeout."""
        # Reset if needed
        if self.needs_reset:
            if time.time() - self.reset_time > RESET_DELAY:
                self._reset()
                self.needs_reset = False
            return

        # Copy control safely
        with self._ctrl_lock:
            ctrl = self.ctrl.copy()

        # Step physics
        with self._physics_lock:
            self.data.ctrl[:] = np.clip(ctrl, -1.0, 1.0)
            mujoco.mj_step(self.model, self.data)
            self.current_step += 1
            ee_pos = self.data.site_xpos[self.ee_site_id].copy()
            qpos = self.data.qpos.copy()
            qvel = self.data.qvel.copy()

        # Compute distance to target
        dist = np.linalg.norm(ee_pos[:2] - self.target[:2])

        # Publish joint states
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = ['shoulder', 'elbow']
        js.position = qpos.tolist()
        js.velocity = qvel.tolist()
        self.joint_pub.publish(js)

        # Log status
        self.get_logger().info(
            f"ep {self.episode_count:3d} | step {self.current_step:3d} | "
            f"dist: {dist:.3f} | "
            f"ee: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}] | "
            f"tgt: [{self.target[0]:.2f}, {self.target[1]:.2f}]",
            throttle_duration_sec=0.5
        )

        # Success & timeout checks
        self.success_count = (self.success_count + 1) if dist < SUCCESS_DIST else 0
        success = self.success_count >= SUCCESS_HOLD
        timeout = self.current_step >= MAX_STEPS

        if success or timeout:
            self.get_logger().info(
                f"{'✅ Success' if success else '⏱ Timeout'} | "
                f"ep {self.episode_count} | dist: {dist:.3f}"
            )
            self.needs_reset = True
            self.reset_time = time.time()


def main(args: Optional[list] = None) -> None:
    """Initialize ROS2 and spin the MuJoCo node."""
    rclpy.init(args=args)
    node = MujocoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()