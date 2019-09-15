import json
import os
import pickle
import sys
from itertools import combinations  # for dimension projection

import numpy as np
import tensorflow as tf

from td3fd import config, logger
from td3fd.util.mpi_util import mpi_average
from td3fd.util.util import set_global_seeds

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class Trainer:
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
        # checkpoint files
        self.ckpt_save_path = os.path.join(self.root_dir, "rl_ckpt")
        os.makedirs(self.ckpt_save_path, exist_ok=True)
        self.ckpt_weight_path = os.path.join(self.ckpt_save_path, "rl_weights.ckpt")
        self.ckpt_rb_path = os.path.join(self.ckpt_save_path, "rb_data.npz")
        self.ckpt_rollout_path = os.path.join(self.ckpt_save_path, "rollout_history.pkl")
        self.ckpt_evaluator_path = os.path.join(self.ckpt_save_path, "evaluator_history.pkl")
        self.ckpt_trainer_path = os.path.join(self.ckpt_save_path, "trainer.pkl")
        # rl (cannot restart training)
        policy_save_path = os.path.join(self.root_dir, "rl")
        os.makedirs(policy_save_path, exist_ok=True)
        best_policy_path = os.path.join(policy_save_path, "policy_best.pkl")
        latest_policy_path = os.path.join(policy_save_path, "policy_latest.pkl")
        periodic_policy_path = os.path.join(policy_save_path, "policy_{}.pkl")
        # queries
        query_shaping_save_path = os.path.join(self.root_dir, "query_shaping")
        os.makedirs(query_shaping_save_path, exist_ok=True)
        query_policy_save_path = os.path.join(self.root_dir, "query_policy")
        os.makedirs(query_policy_save_path, exist_ok=True)

        # Adding demonstration data to the demonstration buffer
        if self.policy.demo_strategy != "none" or self.policy.sample_demo_buffer:
            demo_file = os.path.join(self.root_dir, "demo_data.npz")
            assert os.path.isfile(demo_file), "demonstration training set does not exist"
            self.policy.init_demo_buffer(demo_file, update_stats=self.policy.sample_demo_buffer)

        # Restart-training
        self.best_success_rate = -1
        self.epoch = 0
        self.restart = self._load_states()

        # Pre-Training a potential function (currently we do not support restarting from this state)
        if not self.restart:
            self._train_potential()
            self._train_pure_bc()
            self._store_states()

        # Train the rl agent
        for epoch in range(self.epoch, self.n_epochs):
            # store anything we need into a numpyz file.
            self.policy.query_policy(
                filename=os.path.join(
                    query_policy_save_path, "query_{:03d}.npz".format(epoch)
                ),  # comment to show plot
                fid=3,
            )
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
            self._log(epoch)
            # save the policy
            save_msg = ""
            success_rate = mpi_average(self.evaluator.current_success_rate(), comm=self.comm)
            if self.rank == 0 and success_rate >= self.best_success_rate:
                self.best_success_rate = success_rate
                logger.info("New best success rate: {}.".format(self.best_success_rate))
                self.policy.save_policy(best_policy_path)
                save_msg += "best, "
            if self.rank == 0 and self.save_interval > 0 and epoch % self.save_interval == (self.save_interval - 1):
                policy_path = periodic_policy_path.format(epoch)
                self.policy.save_policy(policy_path)
                self._store_states()
                save_msg += "periodic, "
            if self.rank == 0:
                self.policy.save_policy(latest_policy_path)
                save_msg += "latest"
            logger.info("Saving", save_msg, "policy.")

        self._mpi_sanity_check()

    def _train_potential(self):
        if not self.policy.demo_strategy in ["nf", "gan"]:
            return
        logger.info("Training the policy for reward shaping.")
        for epoch in range(self.shaping_n_epochs):
            loss = self.policy.train_shaping()
            if self.rank == 0 and epoch % (self.shaping_n_epochs / 100) == (self.shaping_n_epochs / 100 - 1):
                logger.info("epoch: {} demo shaping loss: {}".format(epoch, loss))

    def _train_pure_bc(self):
        if self.policy.demo_strategy != "pure_bc":
            return
        logger.info("Training the policy using pure behavior cloning")
        for epoch in range(self.pure_bc_n_epochs):
            loss = self.policy.train_pure_bc()
            if self.rank == 0 and epoch % (self.pure_bc_n_epochs / 100) == (self.pure_bc_n_epochs / 100 - 1):
                logger.info("epoch: {} demo shaping loss: {}".format(epoch, loss))
        self.policy.init_target_net()

    def _log(self, epoch):
        logger.record_tabular("epoch", epoch)
        for key, val in self.evaluator.logs("test"):
            logger.record_tabular(key, mpi_average(val, comm=self.comm))
        for key, val in self.rollout_worker.logs("train"):
            logger.record_tabular(key, mpi_average(val, comm=self.comm))
        for key, val in self.policy.logs():
            logger.record_tabular(key, mpi_average(val, comm=self.comm))
        if self.rank == 0:
            logger.dump_tabular()

    def _store_states(self):
        if self.comm is not None:
            return
        self.policy.save_weights(self.ckpt_weight_path)
        self.policy.save_replay_buffer(self.ckpt_rb_path)
        self.rollout_worker.dump_history_to_file(self.ckpt_rollout_path)
        self.evaluator.dump_history_to_file(self.ckpt_evaluator_path)
        with open(self.ckpt_trainer_path, "wb") as f:
            pickle.dump([self.epoch, self.best_success_rate], f)
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
                self.epoch, self.best_success_rate = pickle.load(f)
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

    def _mpi_sanity_check(self):
        if self.comm is None:
            return
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        self.comm.Bcast(root_uniform, root=0)
        if self.rank != 0:
            assert local_uniform[0] != root_uniform[0]


def main(root_dir, comm=None, **kwargs):

    assert root_dir is not None, "provide root directory for saving training data"

    # Consider rank as pid.
    if comm is None:
        comm = MPI.COMM_WORLD if MPI is not None else None
    num_cpu = comm.Get_size() if comm is not None else 1
    rank = comm.Get_rank() if comm is not None else 0

    if (MPI is None or MPI.COMM_WORLD.Get_rank() == 0) and rank == 0:
        logger.configure(dir=root_dir, format_strs=["stdout", "log", "csv"], log_suffix="")
    elif rank == 0:
        logger.configure(dir=root_dir, format_strs=["log", "csv"], log_suffix="")
    else:
        logger.configure(format_strs=["log"])
    assert logger.get_dir() is not None

    log_level = 2  # 1 for debugging, 2 for info
    logger.set_level(log_level)
    logger.info("Setting log level to {}.".format(log_level))
    logger.info("Launching the training process with {} cpu core(s).".format(num_cpu))

    # since mpi_adam does not support restarting, we set comm to None is only 1 cpu core so policy is trained with the
    # default adam optimizer
    if num_cpu == 1:
        comm = None  # set comm to None so that we can restart training
        logger.info("Changing comm to None since single thread.")

    # Get default params from config and update params.
    param_file = os.path.join(root_dir, "copied_params.json")
    if os.path.isfile(param_file):
        with open(param_file, "r") as f:
            params = json.load(f)
        config.check_params(params)
    else:
        logger.warn("WARNING: params.json not found! using the default parameters.")
        params = config.DEFAULT_PARAMS.copy()
    if rank == 0:
        comp_param_file = os.path.join(root_dir, "params.json")
        with open(comp_param_file, "w") as f:
            json.dump(params, f)

    # Reset default graph (must be called before setting seed)
    tf.reset_default_graph()
    # seed everything.
    set_global_seeds(params["seed"])
    # get a new default session for the current default graph
    tf.InteractiveSession()

    # Prepare parameters for training
    params = config.add_env_params(params=params)

    # Configure and train rl agent
    policy = config.configure_ddpg(params=params, comm=comm)
    rollout_worker = config.config_rollout(params=params, policy=policy)
    evaluator = config.config_evaluator(params=params, policy=policy)

    # Launch the training script
    Trainer(
        comm=comm,
        root_dir=root_dir,
        policy=policy,
        rollout_worker=rollout_worker,
        evaluator=evaluator,
        **params["train"]
    )

    tf.get_default_session().close()


if __name__ == "__main__":

    from td3fd.util.cmd_util import ArgParser

    ap = ArgParser()
    # logging and saving path
    ap.parser.add_argument("--root_dir", help="directory to launching process", type=str, default=None)

    ap.parse(sys.argv)
    main(**ap.get_dict())
