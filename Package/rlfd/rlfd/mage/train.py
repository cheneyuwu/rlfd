import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf
import ray
from ray import tune

from rlfd import config, logger, policies, metrics
from rlfd.mage import mage
from rlfd.mage.params import default_params
from rlfd.utils.util import set_global_seeds


def train(root_dir, params):
  # Check parameters
  config.check_params(params, default_params.parameters)
  # Seed everything.
  set_global_seeds(params["seed"])
  # Setup paths
  policy_save_path = os.path.join(root_dir, "policies")
  os.makedirs(policy_save_path, exist_ok=True)
  initial_policy_path = os.path.join(policy_save_path, "policy_initial.pkl")
  latest_policy_path = os.path.join(policy_save_path, "policy_latest.pkl")
  periodic_policy_path = os.path.join(policy_save_path, "policy_{}.pkl")
  # Tensorboard summary writer
  summary_writer_path = os.path.join(root_dir, "summaries")
  summary_writer = tf.summary.create_file_writer(summary_writer_path,
                                                 flush_millis=10 * 1000)
  summary_writer.set_as_default()
  # Configure agents and drivers.
  config.add_env_params(params=params)
  mage_params = params["ddpg"]
  mage_params.update({
      "dims": params["dims"].copy(),  # agent takes an input observations
      "max_u": params["max_u"],
      "eps_length": params["eps_length"],
      "fix_T": params["fix_T"],
      "gamma": params["gamma"],
      "info": {
          "env_name": params["env_name"],
          "r_scale": params["r_scale"],
          "r_shift": params["r_shift"],
          "eps_length": params["eps_length"],
          "env_args": params["env_args"],
          "gamma": params["gamma"],
      },
  })
  mage_agent = mage.MAGE(**mage_params)
  # Exploration + evaluation drivers
  random_driver = config.config_rollout(
      params=params,
      policy=policies.RandomPolicy(params["dims"]["o"], params["dims"]["g"],
                                   params["dims"]["u"], params["max_u"]))
  expl_driver = config.config_rollout(params=params,
                                      policy=mage_agent.expl_policy)
  eval_driver = config.config_evaluator(params=params,
                                        policy=mage_agent.eval_policy)

  training_metrics = [
      metrics.EnvironmentSteps(),
      metrics.NumberOfEpisodes(),
      metrics.AverageReturnMetric(),
      metrics.AverageEpisodeLengthMetric(),
  ]
  testing_metrics = [
      metrics.EnvironmentSteps(),
      metrics.NumberOfEpisodes(),
      metrics.AverageReturnMetric(),
      metrics.AverageEpisodeLengthMetric(),
  ]

  demo_file = os.path.join(root_dir, "demo_data.npz")

  # Save the initial mage_agent
  mage_agent.before_training_hook(demo_file=demo_file)
  mage_agent.save(initial_policy_path)
  logger.info("Saving initial mage_agent.")

  save_interval = 0
  random_exploration_cycles = params["ddpg"]["random_exploration_cycles"]
  num_epochs = params["ddpg"]["num_epochs"]
  num_cycles = params["ddpg"]["num_cycles"]
  num_batches = params["ddpg"]["num_batches"]

  for _ in range(random_exploration_cycles):
    experiences = random_driver.generate_rollouts(observers=training_metrics)
    mage_agent.store_experiences(experiences)
    mage_agent.update_stats(experiences)

  for epoch in range(num_epochs):
    logger.record_tabular("epoch", epoch)

    # 1 epoch contains multiple cycles of training, 1 time testing, logging
    # and mage_agent saving

    expl_driver.clear_history()
    for cyc in range(num_cycles):

      experiences = expl_driver.generate_rollouts(observers=training_metrics)
      if num_batches != 0:  # mage_agent is being updated
        mage_agent.store_experiences(experiences)
        mage_agent.update_stats(experiences)

      for _ in range(num_batches):
        mage_agent.train()

      if num_batches != 0:  # mage_agent is being updated
        mage_agent.update_target_network()
        mage_agent.clear_n_step_replay_buffer()

      if cyc == num_cycles - 1:
        # update meta parameters
        potential_weight = mage_agent.update_potential_weight()
        logger.info("Current potential weight: ", potential_weight)

    with tf.name_scope("Training"):
      for metric in training_metrics[2:]:
        metric.summarize(step_metrics=training_metrics[:2])
      for key, val in expl_driver.logs("train"):
        logger.record_tabular(key, val)
      for key, val in mage_agent.logs():
        logger.record_tabular(key, val)

    eval_driver.clear_history()
    experiences = eval_driver.generate_rollouts(observers=testing_metrics)

    with tf.name_scope("Testing"):
      for metric in testing_metrics[2:]:
        metric.summarize(step_metrics=training_metrics[:2])
      for key, val in eval_driver.logs("test"):
        logger.record_tabular(key, val)

    logger.dump_tabular()

    # Save the mage_agent periodically.
    save_msg = ""
    if save_interval > 0 and epoch % save_interval == (save_interval - 1):
      policy_path = periodic_policy_path.format(epoch)
      mage_agent.save(policy_path)
      save_msg += "periodic, "
    mage_agent.save(latest_policy_path)
    save_msg += "latest"
    logger.info("Saving", save_msg, "mage_agent.")

    # For ray status updates
    if ray.is_initialized():
      try:
        tune.report()  # ray 0.8.6
      except:
        tune.track.log()  # previous versions