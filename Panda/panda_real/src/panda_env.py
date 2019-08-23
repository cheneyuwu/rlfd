#!/usr/bin/env python

import sys
import subprocess

import rospy
import roslaunch
import moveit_commander
import panda_client as panda
from geometry_msgs.msg import Twist
from franka_msgs.msg import FrankaState

import numpy as np


class FrankaPandaRobotBase(object):

    def __init__(self):

        
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.position_control_launcher = roslaunch.parent.ROSLaunchParent(
                uuid, ["/home/melissa/Workspace/RLProject/Panda/panda_real/launch/panda_moveit.launch"])


        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.velocity_control_launcher = None#subprocess.Popen(["roslaunch"]+ "/home/melissa/Workspace/RLProject/Panda/panda_real/launch/panda_moveit.launch")


        self.velocity_control_launcher = roslaunch.parent.ROSLaunchParent(
                uuid, ["/home/melissa/Workspace/RLProject/Panda/panda_real/launch/franka_arm_vel_controller.launch"])

        self.velocity_publisher = rospy.Publisher(
            "/franka_control/target_velocity", Twist, queue_size=1)

        self.panda_arm_state_sub =  rospy.Subscriber(
            "/franka_state_controller/franka_states",
            FrankaState,
            self.state_callback)

        self.panda_arm_velocity =  rospy.Subscriber(
            "/franka_control/current_velocity",
            Twist,
            self.velocity_callback)

        self.dt = 5.0
        self.rate = rospy.Rate(1.0 / self.dt) # 2hz

        self.cur_pos = None
        self.last_pos = None

        #self.enable_vel_control()
        #self.rate.sleep()

    class ActionSpace:
        def __init__(self, seed=0):
            self.random = np.random.RandomState(seed)
            self.shape = (3,)

        def sample(self):
            return self.random.rand(3)

    def compute_reward(self, achieved_goal, desired_goal, info=0):
        raise NotImplementedError

    def seed(self, seed):
        return
    
    def render(self):
        return

    def close(self):
        self.disable_enable_vel_controlvel_control()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.apply_velocity_action(action)
        self.rate.sleep()
        #return self._get_obs()

    def reset(self):

        self.disable_vel_control()
        self.rate.sleep()

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

        self.enable_vel_control()

        self.last_pos = self.cur_pos
        return self._get_obs()

    def enable_vel_control(self):
        try:
            #self.velocity_control_launcher.start()
            self.velocity_control_launcher = \
            subprocess.Popen(["roslaunch"]+ ["/home/melissa/Workspace/RLProject/Panda/panda_real/launch/panda_moveit.launch"])


        except rospy.ROSInterruptException:
            print('Enable velocity launch failed.')
            return

    def disable_vel_control(self):
        #self.velocity_control_launcher.shutdown()
        self.velocity_control_launcher.terminate()
        #self.velocity_control_launcher.wait()


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

    def _get_obs(self):
        raise NotImplementedError


    def state_callback(self, data):
        self.cur_pos = np.asarray(data.O_T_EE[12:15])
    
    def velocity_callback(self, data):
        pass

class FrankaPegInHole(FrankaPandaRobotBase):
    
    def __init__(self):
        super(FrankaPegInHole, self).__init__()
        self.goal = np.array((0.5, 0.0, 0.36))
        self.threshold = 0.05
        self.sparse = False
    
    def compute_reward(self, achieved_goal, desired_goal, info=0):
        distance = self._compute_distance(achieved_goal, desired_goal)
        if self.sparse == False:
            return -distance
            # return np.maximum(-0.5, -distance)
            # return 0.05 / (0.05 + distance)
        else:  # self.sparse == True
            return -(distance >= self.threshold).astype(np.int64)

    def _compute_distance(self, achieved_goal, desired_goal):
        achieved_goal = achieved_goal.reshape(-1, 3)
        desired_goal = desired_goal.reshape(-1, 3)
        distance = np.sqrt(np.sum(np.square(achieved_goal - desired_goal), axis=1))
        return distance

    def _get_obs(self):
        """
        Get observation
        """
        pos = self.cur_pos
        vel = (pos - self.last_pos) / self.dt
        self.last_pos = pos
        obs = np.concatenate((pos, vel), axis=0)
        ag = obs[:3].copy()
        r = self.compute_reward(ag, self.goal)
        # return distance as metric to measure performance
        distance = self._compute_distance(ag, self.goal)
        # is_success or not
        is_success = distance < self.threshold
        return (
            {"observation": obs, "desired_goal": self.goal, "achieved_goal": ag},
            r,
            0,
            {"is_success": is_success, "shaping_reward": -distance},
        )   

if __name__ == '__main__':
    rospy.init_node("panda_arm_env")
    velocity_publisher = rospy.Publisher(
            "/franka_control/target_velocity", Twist, queue_size=1)
    panda_robo = FrankaPegInHole()
    #panda_robo.reset()
    print('Env reset.')
    panda_robo.enable_vel_control()
    #print('Enable vel control')
    counter = 0
    # Don't send full-speed commands too fast
    # as it drives the controller into a lock state
    x_pos = 1.0
    x_decay = 0.9

    rate = rospy.Rate(5)

    while not rospy.is_shutdown() and counter < 50:
        twist = Twist()
        twist.linear.x = x_pos
        twist.linear.y = 0
        twist.linear.z = 0
        velocity_publisher.publish(twist)

        counter += 1
        x_pos *= x_decay
        rate.sleep()
    
    # panda_robo.close()