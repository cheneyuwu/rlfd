""" Active learning project

    This python script loads a trained rl policy and uses it to navigate the agent in its environment. You only need to
    provide the policy file generated by train*.py and the script will figure out which env should be used.

"""
import os
import numpy as np
import pickle

from yw.util.mpi_util import set_global_seeds
from yw.env.suite_wrapper import make

import robosuite as suite


# DDPG Package import
from yw.tool import logger


def play(policy_file, seed, num_itr, eps_length, env_args, **kwargs):

    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, "rb") as f:
        policy = pickle.load(f)

    # Prepare params.
    env_args = dict(env_args) if env_args != None else policy.info["env_args"]
    eps_length = eps_length if eps_length != 0 else (policy.info["eps_length"] if policy.info["eps_length"] != 0 else policy.T)

    env = make(policy.info["env_name"], **env_args)
    for _ in range(num_itr):
        obs = env.reset()
        o = obs["observation"]
        g = obs["desired_goal"]
        # generate episodes
        for _ in range(eps_length):
            policy_output = policy.get_actions(
                o, 0, g, compute_Q=False, noise_eps=0.0, random_eps=0.0, use_target_net=False
            )
            u = np.array(policy_output)
            obs, _, _, _ = env.step(u)
            o = obs["observation"]
            g = obs["desired_goal"]
            env.render()
            


if __name__ == "__main__":
    import sys
    from yw.util.cmd_util import ArgParser

    ap = ArgParser()

    ap.parser.add_argument("--policy_file", help="demonstration training dataset", type=str, default=None)
    ap.parser.add_argument("--seed", help="RNG seed", type=int, default=413)
    ap.parser.add_argument("--num_itr", help="number of iterations", type=int, default=1)
    ap.parser.add_argument("--eps_length", help="length of each episode", type=int, default=0)
    ap.parser.add_argument(
        "--env_arg",
        help="extra args passed to the environment",
        action="append",
        type=lambda kv: [kv.split(":")[0], eval(str(kv.split(":")[1] + '("' + kv.split(":")[2] + '")'))],
        dest="env_args",
    )
    ap.parse(sys.argv)

    play(**ap.get_dict())