from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import io
import imageio

import gin
import tensorflow as tf

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS


def run_episodes_and_create_video(policy, eval_tf_env, eval_py_env):
  num_episodes = 1
  frames = []
  for _ in range(num_episodes):
    time_step = eval_tf_env.reset()
    frames.append(eval_py_env.render(mode='human'))
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = eval_tf_env.step(action_step.action)
      frames.append(eval_py_env.render(mode='human'))
  # gif_file = io.BytesIO()
  # imageio.mimsave(gif_file, frames, format='gif', fps=60)
  # IPython.display.display(embed_gif(gif_file.getvalue()))


@gin.configurable
def visualize(
    root_dir,
    env_name='HalfCheetah-v2',
):
  eval_py_env = suite_gym.load(env_name)
  eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
  policy_dir = os.path.join(root_dir, 'policy')
  saved_policy = tf.compat.v2.saved_model.load(policy_dir)
  run_episodes_and_create_video(saved_policy, eval_env, eval_py_env)


def main(_):
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  visualize(FLAGS.root_dir)
  print(gin.operative_config_str())


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
