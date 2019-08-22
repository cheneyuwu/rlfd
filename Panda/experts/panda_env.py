#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import panda_client as panda


class FrankaPandaRobot:

    def __init__(self):
        try:
            moveit_commander.roscpp_initialize(sys.argv)
            rospy.init_node('panda_experiment', anonymous=True)

            self.scene = moveit_commander.PlanningSceneInterface()

            self.panda = panda.PandaClient()

        except rospy.ROSInterruptException:
            return

    def step(self, action):
        self.panda.apply_action(action)

    def reset(self):
        self.panda.go_home()

    def enable_grip(self):
        pass

    def disable_grip(self):
        pass


if __name__ == '__main__':
    panda_robo = FrankaPandaRobot()
    panda_robo.reset()
