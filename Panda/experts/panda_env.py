#!/usr/bin/env python

import sys
import rospy
import roslaunch
import moveit_commander
import panda_client as panda


class FrankaPandaRobot:


    def step(self, action):
        self.panda.apply_velocity_action(action)

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
            # rospy.sleep(3)
            # launch.shutdown()

        except rospy.ROSInterruptException:
            return



    def enable_vel_control(self):
        pass

    def enable_grip(self):
        pass

    def disable_grip(self):
        pass


if __name__ == '__main__':
    panda_robo = FrankaPandaRobot()
    panda_robo.reset()

    # while not rospy.is_shutdown():
    #	panda_robo.step([0.0, 0.0, 0.0])
    #    rospy.sleep(0.1)
