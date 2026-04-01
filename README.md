

## Repository Structure

```

.
├── rl_training/
│   ├── train.py            # PPO training script
│   ├── evaluate.py         # Policy evaluation script
│   └── export_model.py     # Save trained model for ROS2
├── ros2_cpp/
│   └── src/
│       └── arm_controller/
│           ├── CMakeLists.txt
│           ├── package.xml
│           └── src/controller_node.cpp   # ROS2 C++ node for policy inference
├── python_sim/
│   └── mujoco_node.py       # Python MuJoCo simulation node
├── env/
│   ├── arm_env.py           # Custom Gym environment
│   └── twolink.xml          # MuJoCo model file
├── requirements.txt         # Python dependencies
├── checkpoint/              # Trained models
└── README.md

````

---

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
````

Dependencies include: `numpy`, `torch`, `gymnasium`, `mujoco`, `tianshou`, `rclpy`, `sensor_msgs`, `std_msgs`.

> **Note:** MuJoCo must already be installed and licensed.

---

### 2. ROS2 Package Setup

1. Create a ROS2 workspace if you don’t already have one:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

2. Copy `ros2_cpp/src/arm_controller` into your workspace:

```bash
cp -r /path/to/ros2_cpp/src/arm_controller .
```

3. Build the workspace:

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

---

## Training the PPO Policy

1. Run the training script:

```bash
python rl_training/train.py
```

2. Save the trained model using `export_model.py`:

```bash
python rl_training/export_model.py 
```

The exported model will be used by the ROS2 C++ controller node.

---

## Running the Simulation & ROS2 Controller
1. **Run the C++ Controller Node**

```bash
ros2 run arm_controller controller_node --ros-args -p model_path:=/path/to/checkpoint/ppo_arm.pth
```

* The node reads joint states and target positions, computes actions using the trained policy, and publishes torque commands.

---

2. **Start the MuJoCo Simulation**

```bash
python python_sim/mujoco_node.py
```

* Publishes:

  * `/joint_states`
  * `/target_pos`
* Subscribes:

  * `/joint_torque_cmd`



