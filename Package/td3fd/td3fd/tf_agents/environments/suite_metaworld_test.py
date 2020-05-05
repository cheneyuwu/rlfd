# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for suite_metaworld."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import gin

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.utils import test_utils

import suite_metaworld

import numpy as np


class SuiteMetaworldTest(test_utils.TestCase):

  def tearDown(self):
    gin.clear_config()
    super(SuiteMetaworldTest, self).tearDown()

  def test_load_adds_time_limit_steps(self):
    env = suite_metaworld.load('button-press_topdown-v1')
    self.assertIsInstance(env, py_environment.PyEnvironment)
    self.assertIsInstance(env, wrappers.TimeLimit)

  # def test_load_disable_step_limit(self):
  #   env = suite_metaworld.load('CartPole-v1', max_episode_steps=0)
  #   self.assertIsInstance(env, py_environment.PyEnvironment)
  #   self.assertNotIsInstance(env, wrappers.TimeLimit)

  # def test_load_disable_wrappers_applied(self):
  #   duration_wrapper = functools.partial(wrappers.TimeLimit, duration=10)
  #   env = suite_metaworld.load('CartPole-v1', max_episode_steps=0, env_wrappers=(duration_wrapper,))
  #   self.assertIsInstance(env, py_environment.PyEnvironment)
  #   self.assertIsInstance(env, wrappers.TimeLimit)

  # def test_custom_max_steps(self):
  #   env = suite_metaworld.load('CartPole-v1', max_episode_steps=5)
  #   self.assertIsInstance(env, py_environment.PyEnvironment)
  #   self.assertIsInstance(env, wrappers.TimeLimit)
  #   self.assertEqual(5, env._duration)

  # def testGinConfig(self):
  #   gin.parse_config_file(test_utils.test_src_dir_path('environments/configs/suite_metaworld.gin'))
  #   env = suite_metaworld.load()
  #   self.assertIsInstance(env, py_environment.PyEnvironment)
  #   self.assertIsInstance(env, wrappers.TimeLimit)


if __name__ == '__main__':
  # test_utils.main()
  from tf_agents.specs import tensor_spec

  env = suite_metaworld.load('button-press_topdown-v1')
  env = tf_py_environment.TFPyEnvironment(env, check_dims=True)
  print(env.action_spec())
  print(env.observation_spec())
  print(env.get_info())

  for _ in range(151):
    action = tensor_spec.sample_bounded_spec(env.action_spec())
    time_step = env.step(action[np.newaxis, ...])
    print(time_step)
    # env.pyenv.envs[0].render("human")
    # env.pyenv.render("human")