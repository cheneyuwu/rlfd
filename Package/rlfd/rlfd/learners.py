import os

import tensorflow as tf
import ray
from ray import tune

from rlfd import config, logger, metrics, policies
from rlfd.utils.util import set_global_seeds


class Learner(object):
  """Off Policy Learner. Hopefully this could be based on ray in the future."""

  def __init__(self, root_dir, get_agent, params):
    # Seed everything.
    set_global_seeds(params["seed"])
    # Configure agents and drivers.
    config.add_env_params(params=params)
    agent_params = params["ddpg"]
    agent_params.update({
        "dims": params["dims"].copy(),
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
    self._agent = get_agent(**agent_params)
    self._random_driver = config.config_rollout(
        params=params,
        policy=policies.RandomPolicy(params["dims"]["o"], params["dims"]["g"],
                                     params["dims"]["u"], params["max_u"]))
    self._expl_driver = config.config_rollout(params=params,
                                              policy=self._agent.expl_policy)
    self._eval_driver = config.config_evaluator(params=params,
                                                policy=self._agent.eval_policy)
    self._training_metrics = [
        metrics.EnvironmentSteps(),
        metrics.NumberOfEpisodes(),
        metrics.AverageReturnMetric(),
        metrics.AverageEpisodeLengthMetric(),
    ]
    self._testing_metrics = [
        metrics.EnvironmentSteps(),
        metrics.NumberOfEpisodes(),
        metrics.AverageReturnMetric(),
        metrics.AverageEpisodeLengthMetric(),
    ]
    # Learning parameters
    self._random_exploration_cycles = params["ddpg"][
        "random_exploration_cycles"]
    self._num_epochs = params["ddpg"]["num_epochs"]
    self._num_cycles = params["ddpg"]["num_cycles"]
    self._num_batches = params["ddpg"]["num_batches"]
    # Setup paths
    self._save_interval = 0
    policy_path = os.path.join(root_dir, "policies")
    os.makedirs(policy_path, exist_ok=True)
    self._initial_policy_path = os.path.join(policy_path, "policy_initial.pkl")
    self._periodic_policy_path = os.path.join(policy_path, "policy_{}.pkl")
    self._latest_policy_path = os.path.join(policy_path, "policy_latest.pkl")
    # Tensorboard summary writer
    summary_writer_path = os.path.join(root_dir, "summaries")
    summary_writer = tf.summary.create_file_writer(summary_writer_path,
                                                   flush_millis=10 * 1000)
    summary_writer.set_as_default()

    demo_file = os.path.join(root_dir, "demo_data.npz")
    # Save the initial agent
    self._agent.before_training_hook(demo_file=demo_file)
    self._agent.save(self._initial_policy_path)
    logger.info("Saving initial agent.")

  def learn(self):

    for _ in range(self._random_exploration_cycles):
      experiences = self._random_driver.generate_rollouts(
          observers=self._training_metrics)
      self._agent.store_experiences(experiences)
      self._agent.update_stats(experiences)

    for epoch in range(self._num_epochs):
      logger.record_tabular("epoch", epoch)

      # 1 epoch contains multiple cycles of training, 1 time testing, logging
      # and self._agent saving

      self._expl_driver.clear_history()
      for cyc in range(self._num_cycles):

        experiences = self._expl_driver.generate_rollouts(
            observers=self._training_metrics)
        if self._num_batches != 0:  # self._agent is being updated
          self._agent.store_experiences(experiences)
          self._agent.update_stats(experiences)

        for _ in range(self._num_batches):
          self._agent.train()

        if self._num_batches != 0:  # agent is being updated
          self._agent.update_target_network()
          self._agent.clear_n_step_replay_buffer()

        if cyc == self._num_cycles - 1:
          # update meta parameters
          potential_weight = self._agent.update_potential_weight()
          logger.info("Current potential weight: ", potential_weight)

      with tf.name_scope("Training"):
        for metric in self._training_metrics[2:]:
          metric.summarize(step_metrics=self._training_metrics[:2])
        for key, val in self._expl_driver.logs("train"):
          logger.record_tabular(key, val)
        for key, val in self._agent.logs():
          logger.record_tabular(key, val)

      self._eval_driver.clear_history()
      experiences = self._eval_driver.generate_rollouts(
          observers=self._testing_metrics)

      with tf.name_scope("Testing"):
        for metric in self._testing_metrics[2:]:
          metric.summarize(step_metrics=self._training_metrics[:2])
        for key, val in self._eval_driver.logs("test"):
          logger.record_tabular(key, val)

      logger.dump_tabular()

      # Save the self._agent periodically.
      save_msg = ""
      if (self._save_interval > 0 and
          epoch % self._save_interval == self._save_interval - 1):
        policy_path = self._periodic_policy_path.format(epoch)
        self._agent.save(policy_path)
        save_msg += "periodic, "
      self._agent.save(self._latest_policy_path)
      save_msg += "latest"
      logger.info("Saving", save_msg, "agent.")

      # For ray status updates
      if ray.is_initialized():
        try:
          tune.report()  # ray 0.8.6
        except:
          tune.track.log()  # previous versions


class OnlineLearner(Learner):
  pass


class OfflineLearner(Learner):
  pass