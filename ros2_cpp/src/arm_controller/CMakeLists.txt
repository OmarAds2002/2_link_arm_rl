cmake_minimum_required(VERSION 3.8)
project(arm_controller)

# ===== C++ =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== ROS2 =====
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# ===== LibTorch =====
# 👉 IMPORTANT: export this in terminal instead (see below)
find_package(Torch REQUIRED)

# ===== Build =====
add_executable(controller_node src/controller_node.cpp)

ament_target_dependencies(controller_node
  rclcpp
  sensor_msgs
  std_msgs
)

target_link_libraries(controller_node
  ${TORCH_LIBRARIES}
)

target_include_directories(controller_node PUBLIC
  ${TORCH_INCLUDE_DIRS}
)

# Needed for torch
set_property(TARGET controller_node PROPERTY CXX_STANDARD 17)

# ===== Install =====
install(TARGETS controller_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()