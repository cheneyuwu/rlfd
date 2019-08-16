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
