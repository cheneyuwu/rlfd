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
"""Suite for loading Gym Environments.

Note we use gym.spec(env_id).make() on gym envs to avoid getting a TimeLimit
wrapper on the environment. OpenAI's TimeLimit wrappers terminate episodes
without indicating if the failure is due to the time limit, or due to negative
agent behaviour. This prevents us from setting the appropriate discount value
for the final step of an episode. To prevent that we extract the step limit
from the environment specs and utilize our TimeLimit wrapper.
"""
import gin
from tf_agents.environments import gym_wrapper, wrappers
from tf_agents.environments.suite_gym import wrap_env

import metaworld
from metaworld.envs.mujoco.env_dict import (HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT)


@gin.configurable
def load(environment_name,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         env_wrappers=(),
         spec_dtype_map=None):
  """Loads the selected environment and wraps it with the specified wrappers.

  Note that by default a TimeLimit wrapper is used to limit episode lengths
  to the default benchmarks defined by the registered environments.

  Args:
    environment_name: Name for the environment to load.
    discount: Discount to use for the environment.
    max_episode_steps: If None the max_episode_steps will be set to the default
      step limit defined in the environment's spec. No limit is applied if set
      to 0 or if there is no max_episode_steps set in the environment's spec.
    gym_env_wrappers: Iterable with references to wrapper classes to use
      directly on the gym environment.
    env_wrappers: Iterable with references to wrapper classes to use on the
      gym_wrapped environment.
    spec_dtype_map: A dict that maps gym specs to tf dtypes to use as the
      default dtype for the tensors. An easy way how to configure a custom
      mapping through Gin is to define a gin-configurable function that returns
      desired mapping and call it in your Gin congif file, for example:
      `suite_gym.load.spec_dtype_map = @get_custom_mapping()`.

  Returns:
    A PyEnvironment instance.
  """

  mtw_envs = {**HARD_MODE_CLS_DICT["train"], **HARD_MODE_CLS_DICT["test"]}
  mtw_args = {**HARD_MODE_ARGS_KWARGS["train"], **HARD_MODE_ARGS_KWARGS["test"]}
  args = mtw_args[environment_name]["args"]
  kwargs = mtw_args[environment_name]["kwargs"]
  kwargs["random_init"] = False  # disable random goal locations
  kwargs["obs_type"] = "with_goal"  # disable random goal locations
  kwargs["rewMode"] = "sparse"  # use sparse reward mode by default

  gym_env = mtw_envs[environment_name](*args, **kwargs)

  if max_episode_steps is None and gym_env.max_path_length is not None:
    max_episode_steps = gym_env.max_path_length

  return wrap_env(gym_env,
                  discount=discount,
                  max_episode_steps=max_episode_steps,
                  gym_env_wrappers=gym_env_wrappers,
                  env_wrappers=env_wrappers,
                  spec_dtype_map=spec_dtype_map)
