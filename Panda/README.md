# Franka Emika Panda guide

## Starting up the robot
- Switch on the controller, visit the url `192.168.131.40/Desk` on the browser to access the Desk App.
- To run a task, unlock the joints first. Before running a task, be sure to hold one hand on one of the emergency stop buttons (ESB) (currently there are two, one ESB next to the controller, one next to the arm).

## Communication Setup
- To connect a third pc to enable communication to the nuc via ethernet, you need to setup correct static ips. For instance, you can set a static ip `192.168.132.90` on nuc, and `192.168.132.95` on the PC. Note: `part3` could also be `131` so the same local network.
- Test to make sure both the pc and nuc can ping each other.
- **Debugging** : Make sure you set the static ip everytime you unplug the ethernet cable. Also if you are using the ethernet to usb extension cable, to unplug, make sure you unplug the usb part, and not just the ethernet cable.

### ROS Node communication
This is for communication between nuc and the pc. See [MultipleMachines](http://wiki.ros.org/ROS/Tutorials/MultipleMachines) and [NetworkSetup](http://wiki.ros.org/ROS/NetworkSetup)

From nuc side:
```
export ROS_IP=192.168.132.90  # current pc ip address
export ROS_MASTER_URI=http://192.168.132.90:11311/
```

From the pc side :
Add the following to your ~/.bashrc

```
export ROS_IP=192.168.132.95  # current pc ip address
export ROS_MASTER_URI=http://192.168.132.90:11311/  # nuc which is running roscore
```

To debug, try running `roswtf` which might give some hints as which part is probablematic.

You can also visualize the robot arm in rviz. Run `rviz rviz`, add `Robotmodel`, inside Global Options, set the frame to `panda_link0`.



## Things to keep in mind
- Only one instance of a `franka::Robot` can connect to the robot. This means, that for example the `franka_joint_state_publisher` cannot run in parallel to the `franka_control_node`. This also implies that you cannot execute the visualization example alongside a separate program running a controller.


## Joystick control
Here we assume, we have franka_ros, and running the example controllers. But you need to actually copy the modified controller from the folder on this repo, `franka_controllers/` and overwrite the existing controller inside the franka_example_controllers you are trying to run.

```
cp franka_controllers/cartesian_velocity_example_controller.cpp /path/to/franka_ros/franka_example_controllers/src/
```

```
roslaunch franka_example_controllers cartesian_velocity_example_controller.launch robot_ip:=192.168.131.40 load_gripper:=false
```

From Panda/input_devices launch - This starts the Joy node:
```
roslaunch logitech.launch
```

Do a `rostopic` to make sure `/joy` messages are coming through. If not, either restart the nodes, or take a look at the `logitech.launch` to make sure all parameters are correct.

The velocity controller is subscribed to `/franka_control/target_velocity`. Now run the Joy to Twist node mapper which will be publishing `Twist` messages to this topic.

```
python input_devices/remap_joy_to_twist.py
```

Do a `rostopic echo /franka_control/target_velocity` to make sure the messages are coming in, this should now move the arm, using the following buttons: with D-pad, up/down moves the arm along the `x-axis`, left/right moves along `y-axis`. Suppress `A` and then up/down moves the arm along the `z-axis`.



