// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/cartesian_velocity_example_controller.h>

#include <array>
#include <cmath>
#include <memory>
#include <string>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace franka_example_controllers {

bool CartesianVelocityExampleController::init(hardware_interface::RobotHW* robot_hardware,
                                              ros::NodeHandle& node_handle) {
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("CartesianVelocityExampleController: Could not get parameter arm_id");
    return false;
  }

  velocity_cartesian_interface_ =
      robot_hardware->get<franka_hw::FrankaVelocityCartesianInterface>();
  if (velocity_cartesian_interface_ == nullptr) {
    ROS_ERROR(
        "CartesianVelocityExampleController: Could not get Cartesian velocity interface from "
        "hardware");
    return false;
  }
  try {
    velocity_cartesian_handle_ = std::make_unique<franka_hw::FrankaCartesianVelocityHandle>(
        velocity_cartesian_interface_->getHandle(arm_id + "_robot"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianVelocityExampleController: Exception getting Cartesian handle: " << e.what());
    return false;
  }

  auto state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR("CartesianVelocityExampleController: Could not get state interface from hardware");
    return false;
  }


  /*
  try {
    auto state_handle = state_interface->getHandle(arm_id + "_robot");

    std::array<double, 7> q_start = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    for (size_t i = 0; i < q_start.size(); i++) {
      if (std::abs(state_handle.getRobotState().q_d[i] - q_start[i]) > 0.1) {
        ROS_ERROR_STREAM(
            "CartesianVelocityExampleController: Robot is not in the expected starting position "
            "for running this example. Run `roslaunch franka_example_controllers "
            "move_to_start.launch robot_ip:=<robot-ip> load_gripper:=<has-attached-gripper>` "
            "first.");
        return false;
      }
    }
   
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianVelocityExampleController: Exception getting state handle: " << e.what());
    return false;
  }
  */

  this->current_velocity.linear.x = 0;
  this->current_velocity.linear.y = 0;
  this->current_velocity.linear.z = 0;
  
  this->current_velocity.angular.x = 0;
  this->current_velocity.angular.y = 0;
  this->current_velocity.angular.z = 0;
  
  this->target_velocity_subscriber =
    node_handle.subscribe("/franka_control/target_velocity", 1, &CartesianVelocityExampleController::target_velocity_callback, this);
  
  this->current_velocity_publisher = node_handle.advertise<geometry_msgs::Twist>("/franka_control/current_velocity", 1000);
  return true;
}

void CartesianVelocityExampleController::target_velocity_callback(const geometry_msgs::Twist::ConstPtr& msg) {
  this->target_velocity = *msg;
}

void CartesianVelocityExampleController::starting(const ros::Time& /* time */) {
  elapsed_time_ = ros::Duration(0.0);
}


  /*  
void CartesianVelocityExampleController::update(const ros::Time&,
                                                const ros::Duration& period) {
  elapsed_time_ += period;

  double time_max = 2.0;
  double v_max = 0.05;
  double angle = M_PI / 4.0;
  double cycle = std::floor(
			    pow(-1.0, (elapsed_time_.toSec() -
				       std::fmod(elapsed_time_.toSec(), time_max)) / time_max));
  
  double v = cycle * v_max / 2.0 * (1.0 - std::cos(2.0 * M_PI / time_max * elapsed_time_.toSec()));
  double v_x = std::cos(angle) * v; 
  double v_z = -std::sin(angle) * v;

  
  std::array<double, 6> command = {{v_x, 0.0, 0.0, 0.0, 0.0, 0.0}};
  velocity_cartesian_handle_->setCommand(command);
}
  */

  
void CartesianVelocityExampleController::update(const ros::Time&,
                                                const ros::Duration& period) {
  elapsed_time_ += period;

  double time_max = 4.0;
  double v_max = 0.15;
  double angle = M_PI / 4.0;

  double cycle = std::floor(
			    pow(-1.0, (elapsed_time_.toSec() -
				       std::fmod(elapsed_time_.toSec(), time_max)) / time_max));

  // Velocity needs to ramp up very slowly and smoothly, otherwise the arm locks
  // due to discontinuities in joint commands
  current_velocity.linear.x = 0.999*current_velocity.linear.x +
    0.001 * v_max/2.0 * target_velocity.linear.x;
  
  current_velocity.linear.y = 0.999*current_velocity.linear.y +
    0.001 * v_max/2.0 * target_velocity.linear.y;
  
  current_velocity.linear.z = 0.999*current_velocity.linear.z +
    0.001 * v_max/2.0 * target_velocity.linear.z;
  
  double v_x = std::cos(angle) * current_velocity.linear.x; 
  double v_y = std::cos(angle) * current_velocity.linear.y; 
  double v_z = std::cos(angle) * current_velocity.linear.z; 

  //std::cerr << v_x << "," << target_velocity.linear.x << std::endl;
  
  std::array<double, 6> command = {{v_x, v_y, v_z, 0.0, 0.0, 0.0}};
  velocity_cartesian_handle_->setCommand(command);

  geometry_msgs::Twist vel_msg;
  vel_msg.linear.x = v_x;
  vel_msg.angular.y = v_y;
  vel_msg.angular.z = v_z;

  current_velocity_publisher.publish(vel_msg);
}
  
  
void CartesianVelocityExampleController::stopping(const ros::Time& /*time*/) {
  // WARNING: DO NOT SEND ZERO VELOCITIES HERE AS IN CASE OF ABORTING DURING MOTION
  // A JUMP TO ZERO WILL BE COMMANDED PUTTING HIGH LOADS ON THE ROBOT. LET THE DEFAULT
  // BUILT-IN STOPPING BEHAVIOR SLOW DOWN THE ROBOT.
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianVelocityExampleController,
                       controller_interface::ControllerBase)
