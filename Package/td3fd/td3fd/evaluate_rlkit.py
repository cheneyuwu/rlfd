from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger

from td3fd.env_manager import EnvManager
from rlkit.envs.wrappers import NormalizedBoxEnv

filename = str(uuid.uuid4())


def main(policy, env_name):

    data = torch.load(policy)
    policy = data["evaluation/policy"]
    # for serializable envs
    # env = data["evaluation/env"]
    # for envs not serializable
    env_manager = EnvManager(env_name=env_name, with_goal=False)
    env = NormalizedBoxEnv(env_manager.get_env())

    print("Policy loaded")
    # move to gpu by default
    set_gpu_mode(True)
    policy.cuda()
    #
    while True:
        path = rollout(env, policy, max_path_length=env.eps_length, render=True,)
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()
