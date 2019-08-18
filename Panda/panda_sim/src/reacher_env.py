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


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    all_equal = True
    if isinstance(goal, list):
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif isinstance(goal, geometry_msgs.msg.PoseStamped):
        return all_close(goal.pose, actual.pose, tolerance)

    elif isinstance(goal, geometry_msgs.msg.Pose):
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True


class PandaReacherEnv(object):
    """PandaReacherEnv"""

    def __init__(self):
        super(PandaReacherEnv, self).__init__()

        # Initialize `moveit_commander`_ and a `rospy` node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('panda_reacher',
                        anonymous=True)

        # `RobotCommander` object is the outer-level interface to
        # the robot.
        robot = moveit_commander.RobotCommander()

        # `PlanningSceneInterface` object is an interface
        # to the world surrounding the robot.
        scene = moveit_commander.PlanningSceneInterface()

        # `MoveGroupCommander` object is an interface
        # to one group of joints.  In this case the group is the joints in the
        # Panda arm so we set ``group_name = panda_arm``.
        # This interface can be used to plan and execute motions on the Panda.
        group_name = "panda_arm"
        group = moveit_commander.MoveGroupCommander(group_name)

        # `DisplayTrajectory` publisher is used later to publish
        # trajectories for RViz to visualize.
        display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20)

        planning_frame = group.get_planning_frame()
        print("============ Reference frame: %s" % planning_frame)

        eef_link = group.get_end_effector_link()
        print("============ End effector: %s" % eef_link)

        group_names = robot.get_group_names()
        print("============ Robot Groups:", robot.get_group_names())

        # Print the entire state of the robot.
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")

        self.object_name = ''
        self.robot = robot
        self.scene = scene
        self.group = group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

    def go_to_joint_state(self):
        """ Plans to a joint goal position.
        """

        # The Panda's zero configuration is at a singularity
        # <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>
        # So move it to a slightly better configuration.
        joint_goal = self.group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -pi/4
        joint_goal[2] = 0
        joint_goal[3] = -pi/2
        joint_goal[4] = 0
        joint_goal[5] = pi/3
        joint_goal[6] = 0

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for
        # the group
        self.group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.group.stop()

        # Test
        current_joints = self.group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)


def main():
    try:
        panda = PandaReacherEnv()
        print(
            "============ Press `Enter` to execute a movement using a joint state goal ...")
        raw_input()
        panda.go_to_joint_state()

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()
