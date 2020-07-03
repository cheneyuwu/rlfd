import os

import tensorflow as tf
import ray
from ray import tune

from rlfd import config, logger, metrics, policies
from rlfd.utils.util import set_global_seeds

osp = os.path


class Learner(object):
  """Off Policy Learner. Hopefully this could be based on ray in the future."""

  def __init__(self, root_dir, get_agent, params):
    # Seed everything.
    set_global_seeds(params["seed"])
    # Configure agents and drivers.
    config.add_env_params(params=params)
    agent_params = params["agent"]
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

    self._random_driver = config.config_driver(
        params["fix_T"],
        params["seed"],
        make_env=params["make_env"],
        policy=policies.RandomPolicy(params["dims"]["o"], params["dims"]["g"],
                                     params["dims"]["u"], params["max_u"]),
        num_steps=params["expl_num_steps_per_cycle"],
        num_episodes=params["expl_num_episodes_per_cycle"])
    self._expl_driver = config.config_driver(
        params["fix_T"],
        params["seed"],
        make_env=params["make_env"],
        policy=self._agent.expl_policy,
        num_steps=params["expl_num_steps_per_cycle"],
        num_episodes=params["expl_num_episodes_per_cycle"])
    self._eval_driver = config.config_driver(
        params["fix_T"],
        params["seed"],
        make_env=params["make_env"],
        policy=self._agent.eval_policy,
        num_steps=params["eval_num_steps_per_cycle"],
        num_episodes=params["eval_num_episodes_per_cycle"])

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
    self._random_exploration_cycles = params["random_expl_num_cycles"]
    self._num_epochs = params["num_epochs"]
    self._num_cycles_per_epoch = params["num_cycles_per_epoch"]
    self._num_batches_per_cycle = params["num_batches_per_cycle"]
    # Setup policy saving
    self._save_interval = 0
    self._policy_path = osp.join(root_dir, "policies")
    os.makedirs(self._policy_path, exist_ok=True)
    # Tensorboard summary writer
    summary_writer_path = osp.join(root_dir, "summaries")
    summary_writer = tf.summary.create_file_writer(summary_writer_path,
                                                   flush_millis=10 * 1000)
    summary_writer.set_as_default()

    demo_file = osp.join(root_dir, "demo_data.npz")
    # Save the initial agent
    self._agent.before_training_hook(demo_file=demo_file)
    self._agent.save(osp.join(self._policy_path, "policy_initial.pkl"))
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
      for cyc in range(self._num_cycles_per_epoch):

        experiences = self._expl_driver.generate_rollouts(
            observers=self._training_metrics)
        if self._num_batches_per_cycle != 0:  # self._agent is being updated
          self._agent.store_experiences(experiences)
          self._agent.update_stats(experiences)

        for _ in range(self._num_batches_per_cycle):
          self._agent.train()

        if self._num_batches_per_cycle != 0:  # agent is being updated
          self._agent.update_target_network()
          self._agent.clear_n_step_replay_buffer()

        if cyc == self._num_cycles_per_epoch - 1:
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
        self._agent.save(
            osp.join(self._policy_path, "policy_{}.pkl".format(epoch)))
        save_msg += "periodic, "
      self._agent.save(osp.join(self._policy_path, "policy_latest.pkl"))
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