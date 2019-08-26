// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <iostream>

#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/gripper.h>

/**
 * @example echo_robot_state.cpp
 * An example showing how to continuously read the robot state.
 */

int main(int argc, char** argv) {
  std::cerr << "Usage: " << argv[0] << " <outfile.txt?> <roll?> <pitch?> <yaw?> <elbow?>" << std::endl;
  std::cerr << "Saves the robot's state to file <outfile.txt> while in Guiding Mode" << std::endl;
  std::cerr << "<roll?> enables the human to guide the end effector (EE) roll angle" << std::endl;
  std::cerr << "<pitch?> enables the human to guide the end effector (EE) pitch angle" << std::endl;
  std::cerr << "<yaw?> enables the human to guide the end effector (EE) yaw angle" << std::endl;
  std::cerr << "<elbow?> enables the human to guide the elbow (EE) joint" << std::endl;
  
  std::string robot_hostname = "192.168.131.40";
  std::string outfilename = "./outfile.txt";
  bool  roll = true;
  bool  pitch = true;
  bool  yaw = true;
    
  if (argc >= 2) {
    outfilename = argv[1];
  }

  if (argc >= 3) {
    roll = atoi(argv[2]);
  }

  if (argc >= 4) {
    pitch = atoi(argv[3]);
  }

  if (argc >= 5) {
    yaw = atoi(argv[4]);
  }
  
 
  try {
    franka::Robot robot(robot_hostname.c_str());
    franka::Gripper gripper(robot_hostname.c_str());
         
    
    std::array<bool, 6> guiding_mode = {{true,    // x  EE
					 true,    // y  EE 
					 true,    // z  EE 
					 roll,    // roll EE
					 pitch,   // pitch EE 
					 yaw}};   // yaw EE

    bool elbow = true;
    
    robot.setGuidingMode(guiding_mode, elbow);
    
    size_t count = 0;
    robot.read([&count, &gripper](const franka::RobotState& robot_state) {
	// Printing to std::cout adds a delay. This is acceptable for a read loop such as this, but
	// should not be done in a control loop.

	// Transforms in libfranka follow a column-major order
	std::array<double, 3> O_P_EE = {{robot_state.O_T_EE[12],
					 robot_state.O_T_EE[13],
					 robot_state.O_T_EE[14]}};


	franka::GripperState gstate = gripper.readOnce();
	
	std::cout << robot_state.O_T_EE[12] << ","
		  << robot_state.O_T_EE[13] << ","
		  << robot_state.O_T_EE[14] << ","
		  << robot_state.O_T_EE[15] << " -- "
		  << gstate.width << "/"
		  << gstate.max_width << "/"
		  << gstate.is_grasped << "/"
		  << gstate.temperature << std::endl;
	
	
	return true;
      });

    std::cout << "Done." << std::endl;
  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}
