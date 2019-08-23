#!/usr/bin/env python

import sys
import rospy
import roslaunch
import moveit_commander
import panda_client as panda
from geometry_msgs.msg import Twist


class FrankaPandaRobot:

    def __init__(self):

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.position_control_launcher = roslaunch.parent.ROSLaunchParent(
                uuid, ["/home/florian/code/RLProject/Panda/panda_real/launch/panda_moveit.launch"])
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.velocity_control_launcher = roslaunch.parent.ROSLaunchParent(
                uuid, ["/home/florian/code/RLProject/Panda/panda_real/launch/franka_arm_vel_controller.launch"])

        self.velocity_publisher = rospy.Publisher(
            "/franka_control/target_velocity", Twist, queue_size=1)

    def step(self, action):
        self.apply_velocity_action(action)

    def reset(self):

        try:
            self.position_control_launcher.start()
            moveit_commander.roscpp_initialize(sys.argv)
            scene = moveit_commander.PlanningSceneInterface()
            panda_robot = panda.PandaClient()
            panda_robot.go_home()
            # May or may not need to sleep - really depends how long it takes
            # to finish execution before shutting down panda moveit!
            # rospy.sleep(3)
            self.position_control_launcher.shutdown()

        except rospy.ROSInterruptException:
            print('Interrupted before completion during reset.')
            return

    def enable_vel_control(self):
        try:
            self.velocity_control_launcher.start()
        except rospy.ROSInterruptException:
            print('Enable velocity launch failed.')
            return

    def disable_vel_control(self):
        self.velocity_control_launcher.shutdown()

    def enable_grip(self):
        pass

    def disable_grip(self):
        pass

    def apply_velocity_action(self, action):

        twist = Twist()
        twist.linear.x = action[0]
        twist.linear.y = action[1]
        twist.linear.z = action[2]
        self.velocity_publisher.publish(twist)


if __name__ == '__main__':
    rospy.init_node("panda_arm_env")
    panda_robo = FrankaPandaRobot()
    panda_robo.reset()
    print('Env reset.')
    panda_robo.enable_vel_control()
    print('Enable vel control')
    counter = 0
    rate = rospy.Rate(2) # 10hz
    # Don't send full-speed commands too fast
    # as it drives the controller into a lock state
    x_pos = 1.0
    x_decay = 0.9

    while not rospy.is_shutdown() and counter < 50:
        panda_robo.step([x_pos, 0.0, 0.0])
        counter += 1
        x_pos *= x_decay
        rate.sleep()
    panda_robo.disable_vel_control()
