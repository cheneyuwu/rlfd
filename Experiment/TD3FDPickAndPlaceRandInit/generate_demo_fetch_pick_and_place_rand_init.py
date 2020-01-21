import pickle
import sys

import numpy as np
import tensorflow as tf

from td3fd.demo_util.generate_demo_fetch import FetchDemoGenerator, store_demo_data
from td3fd.env_manager import EnvManager
from td3fd.util.cmd_util import ArgParser


class PickAndPlaceDemoGenerator(FetchDemoGenerator):
    def __init__(self, env, policy, system_noise_level, render):
        super().__init__(env, policy, system_noise_level, render)

    def generate_pick_place(self, variance_level, sub_opt_level):

        goal_dim = 0
        obj_pos_dim = 10
        obj_rel_pos_dim = 13

        self._reset()
        # move to the above of the object
        self._move_to_object(obj_pos_dim, obj_rel_pos_dim, offset=0.05, gripper_open=True)
        # grab the object
        self._move_to_object(obj_pos_dim, obj_rel_pos_dim, offset=0.0, gripper_open=False)
        # move to the goal
        # sub1 = 0.5 + sub_opt_level
        # sub2 = 0.5 - sub_opt_level
        # assert sub_opt_level <= 0.5
        # assert variance_level <= 0.2
        # if self.num_itr % 2 == 0:
        #     weight = np.array((sub1, sub2, 0.0)) + np.random.normal(scale=variance_level, size=3)
        # elif self.num_itr % 2 == 1:
        #     weight = np.array((sub2, sub1, 0.0)) + np.random.normal(scale=variance_level, size=3)
        # else:
        #     assert False
        # self._move_to_interm_goal(obj_pos_dim, goal_dim, weight)
        self._move_to_goal(obj_pos_dim, goal_dim)
        # open the gripper
        self._move_to_object(obj_pos_dim, obj_rel_pos_dim, offset=0.03, gripper_open=True)
        # stay until the end
        self._stay()

        self.num_itr += 1
        assert self.episode_info[-1]["is_success"]
        return self.episode_obs, self.episode_act, self.episode_rwd, self.episode_done, self.episode_info


def main(policy_file=None, **kwargs):

    # Change the following parameters
    num_itr = 50
    render = False
    env_name = "YWFetchPickAndPlaceRandInit-v0"
    env = EnvManager(env_name=env_name, eps_length=0).get_env()
    system_noise_level = 0.0
    sub_opt_level = 0.0
    variance_level = 0.0

    # Load policy.
    policy = None
    if policy_file is not None:
        # reset default graph every time this function is called.
        tf.reset_default_graph()
        # get a default session for the current graph
        tf.InteractiveSession()
        with open(policy_file, "rb") as f:
            policy = pickle.load(f)

    demo_data_obs = []
    demo_data_acs = []
    demo_data_rewards = []
    demo_data_dones = []
    demo_data_info = []

    generator = PickAndPlaceDemoGenerator(env=env, policy=policy, system_noise_level=system_noise_level, render=render)

    for i in range(num_itr):
        print("Iteration number: ", i)
        episode_obs, episode_act, episode_rwd, episode_done, episode_info = generator.generate_pick_place(
            sub_opt_level=sub_opt_level, variance_level=variance_level
        )
        # print("observation: ", episode_obs)
        # print("action: ", episode_act)
        demo_data_obs.append(episode_obs)
        demo_data_acs.append(episode_act)
        demo_data_rewards.append(episode_rwd)
        demo_data_dones.append(episode_done)
        demo_data_info.append(episode_info)

    store_demo_data(
        T=env.eps_length,
        num_itr=num_itr,
        demo_data_obs=demo_data_obs,
        demo_data_acs=demo_data_acs,
        demo_data_rewards=demo_data_rewards,
        demo_data_dones=demo_data_dones,
        demo_data_info=demo_data_info,
    )


if __name__ == "__main__":
    exp_parser = ArgParser(allow_unknown_args=False)
    exp_parser.parser.add_argument(
        "--policy_file", help="top level directory to store experiment results", type=str, default=None
    )
    exp_parser.parse(sys.argv)
    main(**exp_parser.get_dict())
