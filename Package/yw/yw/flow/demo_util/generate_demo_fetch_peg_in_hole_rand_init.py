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

    def generate_peg_in_hole(self, sub_opt_level, x_var, y_var, z_var):

        goal_dim = 0
        obj_pos_dim = 0

        self._reset()
        # move to the goal
        interm_goal = 0.08 + sub_opt_level + np.random.uniform(0.0, z_var)
        x_var = np.random.uniform(-x_var, 0.0)
        y_var = np.random.uniform(-y_var, y_var)
        self._move_to_goal(obj_pos_dim, goal_dim, offset=np.array((x_var, y_var, interm_goal)))
        interm_goal = 0.08
        self._move_to_goal(obj_pos_dim, goal_dim, offset=np.array((0.0, 0.0, interm_goal)))
        self._move_to_goal(obj_pos_dim, goal_dim)
        # stay until the end
        self._stay()

        self.num_itr += 1
        assert self.episode_info[-1]["is_success"]
        return self.episode_obs, self.episode_act, self.episode_rwd, self.episode_info


def main(policy_file=None, **kwargs):

    # Change the following parameters
    num_itr = 10
    render = True
    env_name = "FetchPegInHoleRandInit-v1"
    env = EnvManager(env_name=env_name, env_args={}, r_scale=1.0, r_shift=0.0, eps_length=40).get_env()
    system_noise_level = 0.0
    sub_opt_level = 0.15

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
    demo_data_info = []

    generator = PegInHoleDemoGenerator(env=env, policy=policy, system_noise_level=system_noise_level, render=render)

    for i in range(num_itr):
        print("Iteration number: ", i)
        episode_obs, episode_act, episode_rwd, episode_info = generator.generate_peg_in_hole(
            sub_opt_level=sub_opt_level, x_var=0.1, y_var=0.05, z_var=0.05
        )
        demo_data_obs.append(episode_obs)
        demo_data_acs.append(episode_act)
        demo_data_rewards.append(episode_rwd)
        demo_data_info.append(episode_info)

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
