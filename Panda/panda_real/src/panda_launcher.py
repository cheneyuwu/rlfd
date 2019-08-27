#!/usr/bin/env python
import rospy
from copy import copy  # for python2
import numpy as np
import os

from yw.env.env_manager import EnvManager
from yw.flow.demo_util.generate_demo_franka import main as gen_demo
from yw.flow.launch import main as train

exp_dir = "/home/melissa/Workspace/RLProject/Experiment/ExpData/Aug27FrankaPegInHole"

# Collect data
# gen_demo(os.path.join(exp_dir, "demo_data.npz"))

# Run the rl training
kwargs = {
    "exp_dir": exp_dir,
    "targets": ["train:" + os.path.join(exp_dir, "training_config.py")],
    "policy_file": None,
}
train(**kwargs)

# Franka env testing
# env_manager = EnvManager("FrankaPegInHole")
# panda_robot = env_manager.get_env()
# # panda_robot = FrankaPegInHole()    
# actions = (
#     [-1.0, -1.0, -1.0],
#     [+1.0, -1.0, -1.0],
#     [+1.0, +1.0, -1.0],
#     [-1.0, +1.0, -1.0],
#     [-1.0, -1.0, +1.0],
#     [+1.0, -1.0, +1.0],
#     [+1.0, +1.0, +1.0],
#     [-1.0, +1.0, +1.0],
# )

# counter = 0
# while not rospy.is_shutdown():
#     for action in actions:
#         panda_robot.reset()
#         for _ in range(30):
#             obs = panda_robot.step(action)
        

# panda_robot.env.disable_vel_control()
# panda_robot.env.enable_pos_control()
# panda_robot.env.panda_client.go_home()
# panda_robot.env.disable_pos_control()
# panda_robot.env.enable_vel_control()