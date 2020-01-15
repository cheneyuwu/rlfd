from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger

filename = str(uuid.uuid4())


def main(policy):
    data = torch.load(policy)
    policy = data["evaluation/policy"]
    env = data["evaluation/env"]
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
