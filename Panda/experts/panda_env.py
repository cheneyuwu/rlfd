#!/usr/bin/env python

import sys
import rospy
import roslaunch
import moveit_commander
import panda_client as panda
from geometry_msgs.msg import Twist

class FrankaPandaRobot:

    def __init__(self):
        self.velocity_publisher = rospy.Publisher(
            "/franka_control/target_velocity", Twist, queue_size=1)

    def step(self, action):
        self.apply_velocity_action(action)

    def reset(self):

        try:
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            launch = roslaunch.parent.ROSLaunchParent(uuid,["/home/florian/code/RLProject/Panda/panda_real/launch/panda_moveit.launch"])
            launch.start()

            moveit_commander.roscpp_initialize(sys.argv)
            rospy.init_node('panda_experiment', anonymous=True)

            scene = moveit_commander.PlanningSceneInterface()

            panda_robot = panda.PandaClient()
            panda_robot.go_home()
			# May or may not need to sleep - really depends how long it takes
			# to finish execution before shutting down panda moveit!
            #rospy.sleep(3)
            launch.shutdown()

        except rospy.ROSInterruptException:
            return



    def enable_vel_control(self):
        try:
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            launch = roslaunch.parent.ROSLaunchParent(uuid,["/home/florian/code/RLProject/Panda/panda_real/launch/franka_arm_vel_controller.launch"])
            launch.start()

			# May or may not need to sleep - really depends how long it takes
			# to finish execution before shutting down panda moveit!
            #rospy.sleep(3)
            #launch.shutdown()

        except rospy.ROSInterruptException:
            return


    def disable_vel_control(self):
        pass

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
    #panda_robo.reset()
    panda_robo.enable_vel_control()

    while not rospy.is_shutdown():
    	panda_robo.step([0.5, 0.0, 0.0])
        rospy.sleep(0.1)
