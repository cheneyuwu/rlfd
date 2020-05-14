import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from rlfd import config, logger
from rlfd.mage import config as mage_config
from rlfd.utils.cmd_util import ArgParser
from rlfd.utils.util import set_global_seeds

# limit gpu memory growth for tensorflow
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices("GPU")
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

try:
  from mpi4py import MPI
except ImportError:
  MPI = None


def train(root_dir, params):

  # Check parameters
  config.check_params(params, mage_config.default_params)

  # Construct...
  save_interval = 0
  demo_strategy = params["ddpg"]["demo_strategy"]
  num_epochs = params["ddpg"]["num_epochs"]
  num_cycles = params["ddpg"]["num_cycles"]
  num_batches = params["ddpg"]["num_batches"]

  # Seed everything.
  set_global_seeds(params["seed"])

  # Setup paths
  policy_save_path = os.path.join(root_dir, "policies")
  os.makedirs(policy_save_path, exist_ok=True)
  initial_policy_path = os.path.join(policy_save_path, "policy_initial.pkl")
  latest_policy_path = os.path.join(policy_save_path, "policy_latest.pkl")
  periodic_policy_path = os.path.join(policy_save_path, "policy_{}.pkl")

  summary_writer_path = os.path.join(root_dir, "summaries")
  summary_writer = tf.summary.create_file_writer(summary_writer_path,
                                                 flush_millis=10 * 1000)
  summary_writer.set_as_default()

  # Construct ...
  config.add_env_params(params=params)
  policy = mage_config.configure_mage(params=params)
  rollout_worker = config.config_rollout(params=params, policy=policy)
  evaluator = config.config_evaluator(params=params, policy=policy)
  # adding demonstration data to the demonstration buffer
  if demo_strategy != "none" or policy.sample_demo_buffer:
    demo_file = os.path.join(root_dir, "demo_data.npz")
    assert os.path.isfile(
        demo_file), "demonstration training set does not exist"
    episode_batch = policy.init_demo_buffer(demo_file)
    if policy.sample_demo_buffer:
      policy.update_stats(episode_batch)

  with tf.summary.record_if(
      lambda: tf.math.equal(policy.td3_training_step % 200, 0)):

    # Train shaping potential
    if demo_strategy in ["nf", "gan"]:
      policy.train_shaping()

    if policy.initialize_with_bc:
      policy.train_bc()
      policy.initialize_target_net()

      # save the policy
      policy.save(initial_policy_path)
      logger.info("Saving initial policy.")

    # Generate some random experiences before training (used by td3 for gym mujoco envs)
    # TODO: hyper paramter -> put in config files
    num_initial_exploration_steps = 5000  # used in mbpo for half-cheetah environment
    for _ in range(num_initial_exploration_steps):
      episode = rollout_worker.generate_rollouts(random=True)
      policy.store_episode(episode)
      policy.update_stats(episode)

    for epoch in range(num_epochs):
      logger.record_tabular("epoch", epoch)

      # 1 epoch contains multiple cycles of training, 1 time testing, logging
      # and policy saving
      with tf.name_scope("Training"):
        # TODO: move these hyper parameters to config files
        cyc_per_model_training = 250  # set by mbpo for 1000 long halfcheetah env
        rollout_worker.clear_history()
        for cyc in range(num_cycles):

          # TODO: add proper training of the model
          if cyc % cyc_per_model_training == 0:
            policy.train_model()

          experiences = rollout_worker.generate_rollouts()
          if num_batches != 0:  # policy is being updated
            policy.store_episode(experiences)
            policy.update_stats(experiences)

          for _ in range(num_batches):
            policy.train()

          if num_batches != 0:  # policy is being updated
            policy.update_target_net()
            policy.clear_n_step_replay_buffer()

          if cyc == num_cycles - 1:
            # update meta parameters
            potential_weight = policy.update_potential_weight()
            logger.info("Current potential weight: ", potential_weight)

        for key, val in rollout_worker.logs("train"):
          logger.record_tabular(key, val)
        for key, val in policy.logs():
          logger.record_tabular(key, val)

      with tf.name_scope("Testing"):
        evaluator.clear_history()
        experiences = evaluator.generate_rollouts()

        for key, val in evaluator.logs("test"):
          logger.record_tabular(key, val)

      logger.dump_tabular()

      # save the policy
      save_msg = ""
      if save_interval > 0 and epoch % save_interval == (save_interval - 1):
        policy_path = periodic_policy_path.format(epoch)
        policy.save(policy_path)
        save_msg += "periodic, "
      policy.save(latest_policy_path)
      save_msg += "latest"
      logger.info("Saving", save_msg, "policy.")


def main(root_dir, **kwargs):

  # allow calling this script using MPI to launch multiple training processes
  # in which case only 1 process should print to stdout
  if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
    logger.configure(dir=root_dir,
                     format_strs=["stdout", "log", "csv"],
                     log_suffix="")
  else:
    logger.configure(dir=root_dir, format_strs=["log", "csv"], log_suffix="")
  assert logger.get_dir() is not None

  # Get default params from config and update params.
  param_file = os.path.join(root_dir, "params.json")
  assert os.path.isfile(param_file), param_file
  with open(param_file, "r") as f:
    params = json.load(f)

  # Launch the training script
  train(root_dir=root_dir, params=params)


if __name__ == "__main__":

  ap = ArgParser()
  # logging and saving path
  ap.parser.add_argument("--root_dir",
                         help="directory to launching process",
                         type=str,
                         default=None)

  ap.parse(sys.argv)
  main(**ap.get_dict())