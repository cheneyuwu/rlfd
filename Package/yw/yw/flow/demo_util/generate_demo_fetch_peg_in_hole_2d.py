import pickle
import sys

import numpy as np
import tensorflow as tf

from yw.flow.demo_util.generate_demo_fetch_policy import FetchDemoGenerator, store_demo_data
from yw.env.env_manager import EnvManager
from yw.util.cmd_util import ArgParser


class PegInHoleDemoGenerator(FetchDemoGenerator):
    def __init__(self, env, policy, system_noise_level, render):
        super().__init__(env, policy, system_noise_level, render)

    def generate_peg_in_hole(self, sub_opt_level, x_var, y_var, z_var, x_bias):

        goal_dim = 0
        obj_pos_dim = 0

        self._reset()
        # move to the goal
        x_var = np.random.uniform(x_bias - x_var, x_bias)
        y_var = np.random.uniform(-y_var, y_var)
        interm_goal = sub_opt_level + np.random.uniform(-z_var, z_var)
        self._move_to_goal(obj_pos_dim, goal_dim, offset=np.array((x_var, y_var, interm_goal)))
        # interm_goal = 0.1
        # self._move_to_goal(obj_pos_dim, goal_dim, offset=np.array((0.0, 0.0, interm_goal)))
        self._move_to_goal(obj_pos_dim, goal_dim)
        # stay until the end
        self._stay()

        self.num_itr += 1
        # assert self.episode_info[-1]["is_success"]
        return self.episode_obs, self.episode_act, self.episode_rwd, self.episode_info

    def generate_peg_in_hole_2(self, height):
        goal_dim = 0
        obj_pos_dim = 0

        self._reset()
        # move to the goal
        self._move_to_goal(obj_pos_dim, goal_dim, offset=np.array((-0.05, 0.0, height)))
        self._move_to_goal(obj_pos_dim, goal_dim)
        # stay until the end
        self._stay()

        self.num_itr += 1
        # assert self.episode_info[-1]["is_success"]
        return self.episode_obs, self.episode_act, self.episode_rwd, self.episode_info


def main(policy_file=None, **kwargs):

    # Change the following parameters
    num_itr = 40
    render = False
    env_name = "FetchPegInHole2DVersion-v1"
    eps_length = 40
    env = EnvManager(env_name=env_name, env_args={}, r_scale=1.0, r_shift=0.0, eps_length=eps_length).get_env()
    # Load policy.
    policy = None
    if policy_file is not None:
        # reset default graph every time this function is called.
        tf.reset_default_graph()
        # get a default session for the current graph
        tf.InteractiveSession()
        with open(policy_file, "rb") as f:
            policy = pickle.load(f)

    system_noise_level = 0.0

    demo_data_obs = []
    demo_data_acs = []
    demo_data_rewards = []
    demo_data_info = []

    generator = PegInHoleDemoGenerator(env=env, policy=policy, system_noise_level=system_noise_level, render=render)

    itr = 0
    for i in range(35):
        print("Iteration number: ", itr)
        episode_obs, episode_act, episode_rwd, episode_info = generator.generate_peg_in_hole_2(height=0.2 / 35.0 * i)
        # change the suggested action!
        for k in range(eps_length + 1):
            episode_act[k][1] = 0.0
        for k in range(14):
            episode_act[k][2] = np.clip(episode_act[k][2] + 1.0, -1.0, 1.0)
        print("observation: ", episode_obs)
        print("action: ", episode_act)
        demo_data_obs.append(episode_obs)
        demo_data_acs.append(episode_act)
        demo_data_rewards.append(episode_rwd)
        demo_data_info.append(episode_info)
        itr += 1

    for i in range(5):
        print("Iteration number: ", itr)
        episode_obs, episode_act, episode_rwd, episode_info = generator.generate_peg_in_hole_2(
            height=0.2 + 0.1 / 5.0 * i
        )
        # change the suggested action!
        for k in range(eps_length + 1):
            episode_act[k][1] = 0.0
        print("observation: ", episode_obs)
        print("action: ", episode_act)
        demo_data_obs.append(episode_obs)
        demo_data_acs.append(episode_act)
        demo_data_rewards.append(episode_rwd)
        demo_data_info.append(episode_info)
        itr += 1

    assert itr == num_itr

    store_demo_data(
        T=env._max_episode_steps,
        num_itr=num_itr,
        demo_data_obs=demo_data_obs,
        demo_data_acs=demo_data_acs,
        demo_data_rewards=demo_data_rewards,
        demo_data_info=demo_data_info,
    )


if __name__ == "__main__":
    exp_parser = ArgParser(allow_unknown_args=False)
    exp_parser.parser.add_argument(
        "--policy_file", help="top level directory to store experiment results", type=str, default=None
    )
    exp_parser.parse(sys.argv)
    main(**exp_parser.get_dict())
