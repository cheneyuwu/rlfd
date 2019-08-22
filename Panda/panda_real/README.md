# Panda Arm Instructions


## Running things individually

See the README inside Panda/

## Build Workspace
```
catkin build or catkin_make build
chmod +x src/panda_arm/src/remap_joy_to_twist.py
```

## Running
```
rosrun panda_arm remap_joy_to_twist.py
```

```
roslaunch panda_arm franka_arm_vel_joy.launch robot_ip:=192.168.131.40 load_gripper:=false
```

```
roslaunch panda_arm franka_arm_pos_joy.launch robot_ip:=192.168.131.40 load_gripper:=false
```



