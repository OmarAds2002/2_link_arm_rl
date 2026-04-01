/*
controller_node.cpp

ROS2 node for controlling a 2-link arm using a pre-trained PyTorch model.
- Subscribes to /joint_states and /target_pos
- Publishes joint torque commands to /joint_torque_cmd
- Computes actions using a loaded TorchScript PPO actor
*/

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include <vector>
#include <array>
#include <cmath>
#include <mutex>

class ControllerNode : public rclcpp::Node
{
public:
    ControllerNode() : Node("controller_node")
    {
        // ── Load pre-trained TorchScript model ─────────────────────────────
        std::string model_path = this->declare_parameter<std::string>(
            "model_path",
            "checkpoint/ppo_actor.pt"  // relative path inside repo
        );

        try {
            module_ = torch::jit::load(model_path);
            module_.eval();
            RCLCPP_INFO(this->get_logger(), "✅ Model loaded");
        } catch (const c10::Error & e) {
            RCLCPP_FATAL(this->get_logger(), "❌ Failed to load model: %s", e.what());
            throw;
        }

        // Default target before receiving from /target_pos
        target_ = {0.8f, 0.3f};
        target_received_ = false;

        // ── Subscriptions ─────────────────────────────────────────────────
        joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&ControllerNode::joint_callback, this, std::placeholders::_1));

        target_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/target_pos", 10,
            std::bind(&ControllerNode::target_callback, this, std::placeholders::_1));

        // ── Publisher ─────────────────────────────────────────────────────
        pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/joint_torque_cmd", 10);

        RCLCPP_INFO(this->get_logger(), "🚀 Controller ready — waiting for /target_pos");
    }

private:
    torch::jit::script::Module module_;

    // Latest target position
    std::array<float, 2> target_;
    bool target_received_;
    std::mutex target_mutex_;

    // ROS2 subscriptions and publisher
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr    joint_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr target_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr    pub_;

    // ── Forward Kinematics ────────────────────────────────────────────
    std::array<float, 2> forward_kinematics(float q1, float q2)
    {
        constexpr float l1 = 0.5f;
        constexpr float l2 = 0.5f;
        return {
            l1 * std::cos(q1) + l2 * std::cos(q1 + q2),
            l1 * std::sin(q1) + l2 * std::sin(q1 + q2)
        };
    }

    // ── Target Callback ───────────────────────────────────────────────
    void target_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() < 2) {
            RCLCPP_WARN(this->get_logger(), "Received /target_pos with < 2 values");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            target_ = {static_cast<float>(msg->data[0]),
                       static_cast<float>(msg->data[1])};
            target_received_ = true;
        }

        RCLCPP_INFO(this->get_logger(), "🎯 New target: [%.2f, %.2f]",
                    msg->data[0], msg->data[1]);
    }

    // ── Joint State Callback ─────────────────────────────────────────
    void joint_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        if (msg->position.size() < 2 || msg->velocity.size() < 2) return;

        float q1  = static_cast<float>(msg->position[0]);
        float q2  = static_cast<float>(msg->position[1]);
        float dq1 = static_cast<float>(msg->velocity[0]);
        float dq2 = static_cast<float>(msg->velocity[1]);

        auto ee = forward_kinematics(q1, q2);

        // Grab latest target safely
        std::array<float, 2> target;
        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            target = target_;
        }

        // Build 8D observation exactly as in training
        std::vector<float> obs = {q1, q2, dq1, dq2, ee[0], ee[1], target[0], target[1]};
        torch::Tensor obs_tensor = torch::tensor(obs).unsqueeze(0);

        torch::Tensor action;
        {
            torch::NoGradGuard no_grad;
            action = module_.forward({obs_tensor}).toTensor().squeeze(0);
        }

        action = torch::clamp(action, -1.0f, 1.0f);

        std_msgs::msg::Float64MultiArray cmd;
        cmd.data = {action[0].item<double>(), action[1].item<double>()};
        pub_->publish(cmd);

        // Throttled logging
        RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 500,
            "action: [%.3f, %.3f] | ee: [%.2f, %.2f] | tgt: [%.2f, %.2f]",
            cmd.data[0], cmd.data[1], ee[0], ee[1], target[0], target[1]);
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ControllerNode>());
    rclcpp::shutdown();
    return 0;
}