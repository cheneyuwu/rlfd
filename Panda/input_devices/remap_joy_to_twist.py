#!/usr/bin/python

import rospy
import subprocess

from sensor_msgs.msg import Joy
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from actionlib_msgs.msg import GoalID

class JoyToTwist:
    def __init__(self):
	# Map buttons to re-adjust the speed.
	# This would be maximum.
	self.speed_setting = 1
        self.cmd_vel_pub = rospy.Publisher("/franka_control/target_velocity", Twist, queue_size=1)
        self.joy_sub = rospy.Subscriber("joy", Joy, self.on_joy)

    def on_joy(self, data):

        left_speed = -data.axes[1] / self.speed_setting # left stick
        right_speed = -data.axes[4] / self.speed_setting # right stick

        # Calculate speed
        linear_vel  = (left_speed + right_speed) / 2.0 # (m/s)
        angular_vel  = (right_speed - left_speed) / 2.0 # (rad/s)

        # Publish Twist
        twist = Twist()
	twist.linear.x = linear_vel

	# Suppress 'A' to switch motion mode
	if data.buttons[0] == 1 :
	    # Move up/down
	    twist.linear.z = data.axes[1]
	else:
	    # Move left/right
	    twist.linear.y = data.axes[0]

        twist.angular.z = angular_vel
	
        self.cmd_vel_pub.publish(twist)

def main():
    rospy.init_node("joy_to_twist")
    controller = JoyToTwist()
    while not rospy.is_shutdown():
	rospy.sleep(0.5)

if __name__ == '__main__':
    main()
