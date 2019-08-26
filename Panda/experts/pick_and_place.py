#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import actionlib
import franka_gripper
import franka_gripper.msg
from panda_client import PandaClient


def main():
    try:

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('panda_experiment', anonymous=True)

        # This is an interface to the world surrounding the robot
        scene = moveit_commander.PlanningSceneInterface()

        # This is an interface to the robot
        panda_client = PandaClient()

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = 1.0
        pose_goal.orientation.y = 0.0
        pose_goal.orientation.z = 0.0
        pose_goal.orientation.w = 0.0
        pose_goal.position.x = 0.36
        pose_goal.position.y = 0.0
        pose_goal.position.z = 0.37

        panda_client.get_ee_state()

        print "============ Press `1` to move the arm to a fixed valid position..."
        inp = raw_input()
        if inp == '1':
            panda_client.move_ee_to(pose_goal)

        print "============ Press `1` to shift the EE by a delta..."
        inp = raw_input()
        if inp == '1':
            panda_client.shift_ee_by(axis=0, value=-0.05)

        print "============ Press `1` to move the arm to home configuration..."
        inp = raw_input()
        if inp == '1':
            panda_client.move_joints_to(panda_client.home_pose_joint_values)

        print "============ Press `1` to move the gripper ..."
        inp = raw_input()
        if inp == '1':
            panda_client.actionlib_move_gripper_to(width=0.06)

        print "============ Press `1` to grasp ..."
        inp = raw_input()
        if inp == '1':
            panda_client.actionlib_gripper_grasp(width=0.3)

        print "============ Press `1` to put the gripper in the home position..."
        inp = raw_input()
        if inp == '1':
            panda_client.actionlib_gripper_homing()

        print "============ Python tutorial demo complete!"
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()
