import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from td3fd import logger
from yw.ddpg_main import config
from td3fd.util.cmd_util import ArgParser
from td3fd.util.util import set_global_seeds

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class train:
    def __init__(
        self,
        comm,
        root_dir,
        save_interval,
        policy,
        rollout_worker,
        evaluator,
        shaping_n_epochs,
        pure_bc_n_epochs,
        n_epochs,
        n_batches,
        n_cycles,
        **kwargs
    ):
        # Params
        self.comm = comm
        self.rank = comm.Get_rank() if comm is not None else 0
        self.root_dir = root_dir
        self.save_interval = save_interval
        self.policy = policy
        self.rollout_worker = rollout_worker
        self.evaluator = evaluator
        self.shaping_n_epochs = shaping_n_epochs
        self.pure_bc_n_epochs = pure_bc_n_epochs
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        self.n_cycles = n_cycles

        # Setup paths
        # # checkpoint files
        # self.ckpt_save_path = os.path.join(self.root_dir, "rl_ckpt")
        # os.makedirs(self.ckpt_save_path, exist_ok=True)
        # self.ckpt_weight_path = os.path.join(self.ckpt_save_path, "rl_weights.ckpt")
        # self.ckpt_rb_path = os.path.join(self.ckpt_save_path, "rb_data.npz")
        # self.ckpt_rollout_path = os.path.join(self.ckpt_save_path, "rollout_history.pkl")
        # self.ckpt_evaluator_path = os.path.join(self.ckpt_save_path, "evaluator_history.pkl")
        # self.ckpt_trainer_path = os.path.join(self.ckpt_save_path, "trainer.pkl")
        # rl (cannot restart training)
        policy_save_path = os.path.join(self.root_dir, "policies")
        os.makedirs(policy_save_path, exist_ok=True)
        best_policy_path = os.path.join(policy_save_path, "policy_best.pkl")
        latest_policy_path = os.path.join(policy_save_path, "policy_latest.pkl")
        periodic_policy_path = os.path.join(policy_save_path, "policy_{}.pkl")
        # # queries
        # query_shaping_save_path = os.path.join(self.root_dir, "query_shaping")
        # os.makedirs(query_shaping_save_path, exist_ok=True)
        # query_policy_save_path = os.path.join(self.root_dir, "query_policy")
        # os.makedirs(query_policy_save_path, exist_ok=True)

        # Adding demonstration data to the demonstration buffer
        if self.policy.demo_strategy != "none" or self.policy.sample_demo_buffer:
            demo_file = os.path.join(self.root_dir, "demo_data.npz")
            assert os.path.isfile(demo_file), "demonstration training set does not exist"
            self.policy.init_demo_buffer(demo_file, update_stats=self.policy.sample_demo_buffer)

        # Restart-training
        self.best_success_rate = -1
        self.epoch = 0
        # self.restart = self._load_states()

        # Pre-Training a potential function (currently we do not support restarting from this state)
        # if not self.restart:
        # self._train_potential()
        # self._train_pure_bc()
        # self._store_states()
        if not self.policy.demo_strategy in ["nf", "gan"]:
            return
        logger.info("Training the policy for reward shaping.")
        for epoch in range(self.shaping_n_epochs):
            loss = self.policy.train_shaping()
            if self.rank == 0 and epoch % (self.shaping_n_epochs / 100) == (self.shaping_n_epochs / 100 - 1):
                logger.info("epoch: {} demo shaping loss: {}".format(epoch, loss))

        # Train the rl agent
        for epoch in range(self.epoch, self.n_epochs):
            # # store anything we need into a numpyz file.
            # self.policy.query_policy(
            #     filename=os.path.join(
            #         query_policy_save_path, "query_{:03d}.npz".format(epoch)
            #     ),  # comment to show plot
            #     fid=3,
            # )
            # train
            if self.policy.demo_strategy != "pure_bc":
                self.rollout_worker.clear_history()
                for _ in range(self.n_cycles):
                    episode = self.rollout_worker.generate_rollouts()
                    self.policy.store_episode(episode)
                    for _ in range(self.n_batches):
                        self.policy.train()
                    self.policy.check_train()
                    self.policy.update_target_net()
            # test
            self.evaluator.clear_history()
            self.evaluator.generate_rollouts()

            self.epoch = epoch + 1
            # log
            # self._log(epoch)
            logger.record_tabular("epoch", epoch)
            for key, val in self.evaluator.logs("test"):
                logger.record_tabular(key, val)
            for key, val in self.rollout_worker.logs("train"):
                logger.record_tabular(key, val)
            for key, val in self.policy.logs():
                logger.record_tabular(key, val)
            if self.rank == 0:
                logger.dump_tabular()
            # save the policy
            save_msg = ""
            success_rate = self.evaluator.current_success_rate()
            if self.rank == 0 and success_rate >= self.best_success_rate:
                self.best_success_rate = success_rate
                logger.info("New best success rate: {}.".format(self.best_success_rate))
                self.policy.save_policy(best_policy_path)
                save_msg += "best, "
            if self.rank == 0 and self.save_interval > 0 and epoch % self.save_interval == (self.save_interval - 1):
                policy_path = periodic_policy_path.format(epoch)
                self.policy.save_policy(policy_path)
                # self._store_states()
                save_msg += "periodic, "
            if self.rank == 0:
                self.policy.save_policy(latest_policy_path)
                save_msg += "latest"
            logger.info("Saving", save_msg, "policy.")

def main(root_dir, **kwargs):

    assert root_dir is not None, "provide root directory for saving training data"
    # allow calling this script using MPI to launch multiple training processes, in which case only 1 process should
    # print to stdout
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure(dir=root_dir, format_strs=["stdout", "log", "csv"], log_suffix="")
    else:
        logger.configure(dir=root_dir, format_strs=["log", "csv"], log_suffix="")
    assert logger.get_dir() is not None

    # TODO: delete this
    comm = None

    # Get default params from config and update params.
    param_file = os.path.join(root_dir, "params.json")
    assert os.path.isfile(param_file), param_file
    with open(param_file, "r") as f:
        params = json.load(f)
    config.check_params(params)

    # seed everything.
    set_global_seeds(params["seed"])
    # get a new default session for the current default graph
    tf.compat.v1.InteractiveSession()

    # Prepare parameters for training
    params = config.add_env_params(params=params)

    # Configure and train rl agent
    policy = config.configure_ddpg(params=params, comm=comm)
    rollout_worker = config.config_rollout(params=params, policy=policy)
    evaluator = config.config_evaluator(params=params, policy=policy)

    # Launch the training script
    train(
        comm=comm,
        root_dir=root_dir,
        policy=policy,
        rollout_worker=rollout_worker,
        evaluator=evaluator,
        **params["train"]
    )

    tf.get_default_session().close()


if __name__ == "__main__":

    ap = ArgParser()
    # logging and saving path
    ap.parser.add_argument("--root_dir", help="directory to launching process", type=str, default=None)

    ap.parse(sys.argv)
    main(**ap.get_dict())
