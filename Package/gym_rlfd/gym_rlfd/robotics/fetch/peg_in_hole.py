import os
from gym import utils
from gym_rlfd.robotics import fetch_peginhole_env as fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "peg_in_hole.xml")


class FetchPegInHoleEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="sparse", rand_init=False, no_y=False, pixel=False):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            # add more objects in below as well as in the xml file (2 places), also change the number below
            # 'peg:joint': [1.25, 0.75 ,0.5, 1., 0., 0., 0.],
            "hole:joint": [
                1.45,
                0.75,
                0.5,
                1.0,
                0.0,
                0.0,
                0.0,
            ],  # the 3rd entry is set to a different value to avoid error
        }
        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            n_substeps=20,
            gripper_extra_height=0.2,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            rand_init=rand_init,
            no_y=no_y,
            pixel=pixel,
        )
        utils.EzPickle.__init__(self)
