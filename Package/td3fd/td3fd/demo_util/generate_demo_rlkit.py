import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

from rlkit.core import logger
from rlkit.torch.pytorch_util import set_gpu_mode
from td3fd.util.util import set_global_seeds

from td3fd.env_manager import EnvManager
from rlkit.envs.wrappers import NormalizedBoxEnv

DEFAULT_PARAMS = {
    "seed": 0,
    "num_eps": 40,
    "demo": {"random_eps": 0.0, "noise_eps": 0.1, "render": False},
    "filename": "demo_data.npz",
}


def add_noise_to_action(env, a, noise_eps, random_eps):
    # add noise to a
    gauss_noise = noise_eps * np.random.randn(*a.shape)
    a = a + gauss_noise
    a = np.clip(a, -1.0, 1.0)
    # add random noise to a
    random_act = env.action_space.sample()
    a += np.random.binomial(1, random_eps) * (random_act - a)
    return a


def rollout(env, agent, max_path_length=np.inf, render=False, render_kwargs=None, noise_eps=0.0, random_eps=0.0):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {"mode": "human"}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        a = add_noise_to_action(env, a, noise_eps=noise_eps, random_eps=random_eps)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack((observations[1:, :], np.expand_dims(next_o, 0)))
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def main(policy, root_dir, env_name, **kwargs):
    # Get default params from config and update params.
    param_file = os.path.join(root_dir, "demo_config.json")
    if os.path.isfile(param_file):
        with open(param_file, "r") as f:
            params = json.load(f)
    else:
        print("WARNING: demo_config.json not found! using the default parameters.")
        params = DEFAULT_PARAMS.copy()
        param_file = os.path.join(root_dir, "demo_config.json")
        with open(param_file, "w") as f:
            json.dump(params, f)

    # Set random seed for the current graph
    set_global_seeds(params["seed"])

    # Load policy.
    data = torch.load(policy)
    policy = data["evaluation/policy"]
    # for serializable envs
    # env = data["evaluation/env"]
    # for envs not serializable
    env_manager = EnvManager(env_name=env_name, with_goal=False)
    env = NormalizedBoxEnv(env_manager.get_env())
    print("Policy loaded")
    # comment the next two lines to not use cuda
    set_gpu_mode(True)
    policy.cuda()
    paths = []
    for _ in range(params["num_eps"]):
        path = rollout(env, policy, max_path_length=env.eps_length, **params["demo"])
        paths.append(path)
        print("Cumulative reward of this episode: {}".format(path["rewards"].sum()))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()

    # Store demonstration data (only the main thread)
    os.makedirs(root_dir, exist_ok=True)
    file_name = os.path.join(root_dir, params["filename"])
    pickle.dump(paths, open(file_name, "wb"))
    print("Demo file has been stored into {}.".format(file_name))


if __name__ == "__main__":

    from td3fd.util.cmd_util import ArgParser

    ap = ArgParser()
    ap.parser.add_argument("--root_dir", help="policy store directory", type=str, default=None)
    ap.parser.add_argument("--policy", help="input policy file for training", type=str, default=None)
    ap.parse(sys.argv)

    main(**ap.get_dict())
