import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from td3fd import logger
from td3fd import config
from td3fd.ddpg import config as ddpg_config
from td3fd.util.cmd_util import ArgParser
from td3fd.util.util import set_global_seeds

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class Trainer:
    def __init__(self, root_dir, params):
        # Params
        self.root_dir = root_dir

        # Configure and train rl agent
        self.policy = ddpg_config.configure_ddpg(params=params)
        self.rollout_worker = config.config_rollout(params=params, policy=self.policy)
        self.evaluator = config.config_evaluator(params=params, policy=self.policy)

        self.save_interval = 10
        self.shaping_num_epochs = self.policy.shaping_params["num_epochs"]
        self.num_epochs = self.policy.num_epochs
        self.num_batches = self.policy.num_batches
        self.num_cycles = self.policy.num_cycles
        # Setup paths
        # checkpoint files
        self.ckpt_save_path = os.path.join(self.root_dir, "rl_ckpt")
        os.makedirs(self.ckpt_save_path, exist_ok=True)
        self.ckpt_weight_path = os.path.join(self.ckpt_save_path, "rl_weights.ckpt")
        self.ckpt_rb_path = os.path.join(self.ckpt_save_path, "rb_data.npz")
        self.ckpt_rollout_path = os.path.join(self.ckpt_save_path, "rollout_history.pkl")
        self.ckpt_evaluator_path = os.path.join(self.ckpt_save_path, "evaluator_history.pkl")
        self.ckpt_trainer_path = os.path.join(self.ckpt_save_path, "trainer.pkl")
        # rl policies (cannot restart training)
        policy_save_path = os.path.join(self.root_dir, "rl")
        os.makedirs(policy_save_path, exist_ok=True)
        latest_policy_path = os.path.join(policy_save_path, "policy_latest.pkl")
        periodic_policy_path = os.path.join(policy_save_path, "policy_{}.pkl")

        # Adding demonstration data to the demonstration buffer
        if self.policy.demo_strategy != "none" or self.policy.sample_demo_buffer:
            demo_file = os.path.join(self.root_dir, "demo_data.npz")
            assert os.path.isfile(demo_file), "demonstration training set does not exist"
            self.policy.init_demo_buffer(demo_file, update_stats=self.policy.sample_demo_buffer)

        # if self.policy.demo_strategy in ["nf", "gan"]:
        #     # Load policy.
        #     with open("demo_policy.pkl", "rb") as f:
        #         demo_policy = pickle.load(f)

        # Restart-training
        self.epoch = 0
        self.restart = self._load_states()

        # Pre-Training a potential function (currently we do not support restarting from this state)
        if not self.restart:
            self._train_potential()
            self._store_states()

        # Train the rl agent
        for epoch in range(self.epoch, self.num_epochs):
            # train
            self.rollout_worker.clear_history()
            for _ in range(self.num_cycles):
                episode = self.rollout_worker.generate_rollouts()
                self.policy.store_episode(episode)
                for _ in range(self.num_batches):
                    self.policy.train()
                self.policy.update_target_net()
            # test
            self.evaluator.clear_history()
            episode = self.evaluator.generate_rollouts()
            # if self.policy.demo_strategy in ["nf", "gan"]:
            #     o = episode["o"][:, :-1, ...].reshape(-1, *self.policy.dimo)
            #     g = episode["g"].reshape(-1, *self.policy.dimg)
            #     u = demo_policy.get_actions(o, g, compute_q=False)
            #     u = u.reshape(episode["u"].shape)
            #     episode["u"] = u
            #     self.policy.add_to_demo_buffer(episode)
            #     self._train_potential()

            self.epoch = epoch + 1

            # log
            self._log(epoch)

            # save the policy
            save_msg = ""
            success_rate = self.evaluator.current_success_rate()
            logger.info("Current success rate: {}".format(success_rate))
            if self.save_interval > 0 and epoch % self.save_interval == (self.save_interval - 1):
                policy_path = periodic_policy_path.format(epoch)
                self.policy.save_policy(policy_path)
                self._store_states()
                save_msg += "periodic, "
            self.policy.save_policy(latest_policy_path)
            save_msg += "latest"
            logger.info("Saving", save_msg, "policy.")

    def _train_potential(self):
        if not self.policy.demo_strategy in ["nf", "gan"]:
            return
        logger.info("Training the policy for reward shaping.")
        for epoch in range(self.shaping_num_epochs):
            loss = self.policy.train_shaping()
            if epoch % (self.shaping_num_epochs / 100) == (self.shaping_num_epochs / 100 - 1):
                logger.info("epoch: {} demo shaping loss: {}".format(epoch, loss))
                self.policy.evaluate_shaping()

    def _log(self, epoch):
        logger.record_tabular("epoch", epoch)
        for key, val in self.evaluator.logs("test"):
            logger.record_tabular(key, val)
        for key, val in self.rollout_worker.logs("train"):
            logger.record_tabular(key, val)
        for key, val in self.policy.logs():
            logger.record_tabular(key, val)
        logger.dump_tabular()

    def _store_states(self):
        self.policy.save_weights(self.ckpt_weight_path)
        self.policy.save_replay_buffer(self.ckpt_rb_path)
        self.rollout_worker.dump_history_to_file(self.ckpt_rollout_path)
        self.evaluator.dump_history_to_file(self.ckpt_evaluator_path)
        with open(self.ckpt_trainer_path, "wb") as f:
            pickle.dump(self.epoch, f)
        logger.info("Saving periodic check point.")

    def _load_states(self):
        restart = False
        restart_msg = ""
        if os.path.exists(os.path.join(self.ckpt_save_path, "checkpoint")):
            self.policy.load_weights(self.ckpt_weight_path)
            restart = True
            restart_msg += "rl weights, "
        if os.path.exists(self.ckpt_rb_path):
            self.policy.load_replay_buffer(self.ckpt_rb_path)
            restart_msg += "replay buffer, "
        if os.path.exists(self.ckpt_rollout_path):
            self.rollout_worker.load_history_from_file(self.ckpt_rollout_path)
            restart_msg += "rollout history, "
        if os.path.exists(self.ckpt_evaluator_path):
            self.evaluator.load_history_from_file(self.ckpt_evaluator_path)
            restart_msg += "evaluator history, "
        if os.path.exists(self.ckpt_trainer_path):
            with open(self.ckpt_trainer_path, "rb") as f:
                self.epoch = pickle.load(f)
            restart_msg += "weight"
        # rewrite the progress.csv if necessary
        progress = os.path.join(self.root_dir, "progress.csv")
        self._handle_progress(progress)
        if restart_msg != "":
            logger.info("Loading", restart_msg, "and restarting the training process.")
        return restart

    def _handle_progress(self, progress):
        if not os.path.exists(progress):
            return
        with open(progress, "r") as f:
            f.seek(0)
            lines = f.read().splitlines()
            if not lines:
                return
            keys = lines[0].split(",")
            assert not "" in keys
            f.seek(0)
            lines = f.readlines()
        with open(progress, "w") as f:
            f.write(lines[0])
            for line in lines[1:]:
                if int(line.split(",")[0]) < self.epoch:
                    f.write(line)
            f.flush()


def main(root_dir, **kwargs):

    assert root_dir is not None, "provide root directory for saving training data"
    # allow calling this script using MPI to launch multiple training processes, in which case only 1 process should
    # print to stdout
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure(dir=root_dir, format_strs=["stdout", "log", "csv"], log_suffix="")
    else:
        logger.configure(dir=root_dir, format_strs=["log", "csv"], log_suffix="")
    assert logger.get_dir() is not None

    # Get default params from config and update params.
    param_file = os.path.join(root_dir, "copied_params.json")
    if os.path.isfile(param_file):
        with open(param_file, "r") as f:
            params = json.load(f)
        config.check_params(params)
    else:
        logger.warn("WARNING: params.json not found! using the default parameters.")
        params = ddpg_config.DEFAULT_PARAMS.copy()
    comp_param_file = os.path.join(root_dir, "params.json")
    with open(comp_param_file, "w") as f:
        json.dump(params, f)

    # reset default graph (must be called before setting seed)
    tf.reset_default_graph()
    # seed everything.
    set_global_seeds(params["seed"])
    # get a new default session for the current default graph
    tf.InteractiveSession()

    # Prepare parameters for training
    params = config.add_env_params(params=params)

    # Launch the training script
    Trainer(root_dir=root_dir, params=params)

    tf.get_default_session().close()


if __name__ == "__main__":

    ap = ArgParser()
    # logging and saving path
    ap.parser.add_argument("--root_dir", help="directory to launching process", type=str, default=None)

    ap.parse(sys.argv)
    main(**ap.get_dict())
