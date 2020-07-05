import os

import numpy as np
import gym
import gym.wrappers
from gym import spaces

import d4rl
import gym_rlfd
from rlfd.envs import reacher_2d

try:
  import dmc2gym as dmc
except ModuleNotFoundError:
  dmc = None

try:
  from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT

  mtw_envs = {**HARD_MODE_CLS_DICT["train"], **HARD_MODE_CLS_DICT["test"]}
  mtw_args = {**HARD_MODE_ARGS_KWARGS["train"], **HARD_MODE_ARGS_KWARGS["test"]}

except:
  mtw_envs = None
  mtw_args = None


class EnvWrapper:
  """Wrapper of the environment that does the following:
    1. adjust rewards: r = (r + r_shift) / r_scale
    2. modify state to contain: observation, achieved_goal, desired_goal
  """

  def __init__(self, make_env, r_scale, r_shift):
    self.env = make_env()
    self.r_scale = r_scale
    self.r_shift = r_shift
    if "_max_episode_steps" in self.env.__dict__:
      self.eps_length = self.env._max_episode_steps
    elif "max_path_length" in self.env.__dict__:
      self.eps_length = self.env.max_path_length
    else:
      raise RuntimeError("max episode length unknown.")
    if isinstance(self.env, gym.wrappers.TimeLimit):
      self.env = self.env.env
      env_ = self.env
      while isinstance(env_, gym.Wrapper):
        if isinstance(env_, gym.wrappers.TimeLimit):
          raise ValueError("Can remove only top-level TimeLimit gym.Wrapper.")
          env_ = env_.env
    # need the following properties
    self.action_space = self.env.action_space
    self.observation_space = self.env.observation_space
    self.max_u = self.action_space.high[0]

  def reset(self, **kwargs):
    state = self.env.reset(**kwargs)
    return self._transform_state(state)

  def get_dataset(self):
    """This is for d4rl environments only"""
    # observations, actions, rewards, terminals
    try:
      dataset = self.env.get_dataset()
      dataset = dict(o=dataset["observations"][:-1],
                     o_2=dataset["observations"][1:],
                     u=dataset["actions"][:-1],
                     r=dataset["rewards"][:-1].reshape((-1, 1)),
                     done=dataset["terminals"][:-1].reshape((-1, 1)))
    except AttributeError:
      dataset = None
    return dataset

  def render(self, **kwargs):
    return self.env.render(**kwargs)

  def seed(self, seed=0):
    return self.env.seed(seed)

  def step(self, action):
    state, r, done, info = self.env.step(action)
    r = (r + self.r_shift) / self.r_scale
    return self._transform_state(state), r, done, info

  def close(self):
    return self.env.close()


class GoalEnvWrapper(EnvWrapper):

  def __init__(self, make_env, r_scale, r_shift):
    super().__init__(make_env, r_scale, r_shift)

    if not type(self.observation_space) == spaces.Dict:
      self.observation_space = spaces.Dict(
          dict(
              observation=self.env.observation_space,
              desired_goal=spaces.Box(-np.inf,
                                      np.inf,
                                      shape=np.empty(0).shape,
                                      dtype="float32"),
              achieved_goal=spaces.Box(-np.inf,
                                       np.inf,
                                       shape=np.empty(0).shape,
                                       dtype="float32"),
          ))

  def _transform_state(self, state):
    """
    modify state to contain: observation, achieved_goal, desired_goal
    """
    if not type(state) == dict:
      state = {
          "observation": state,
          "achieved_goal": np.empty(0),
          "desired_goal": np.empty(0)
      }
    return state

  def get_dataset(self):
    """This is for d4rl environments only"""
    # observations, actions, rewards, terminals
    dataset = super().get_dataset()
    if dataset:
      dataset["g"] = np.empty((dataset["o"].shape[0], 0))
      dataset["g_2"] = np.empty((dataset["o"].shape[0], 0))
      dataset["ag"] = np.empty((dataset["o"].shape[0], 0))
      dataset["ag_2"] = np.empty((dataset["o"].shape[0], 0))
    return dataset


class NoGoalEnvWrapper(EnvWrapper):

  def __init__(self, make_env, r_scale, r_shift):
    super().__init__(make_env, r_scale, r_shift)

    if type(self.observation_space) is spaces.Dict:
      assert len(self.observation_space["desired_goal"].low.shape) == 1
      assert len(self.observation_space["observation"].low.shape) == 1
      shape = (self.observation_space["observation"].high.shape[0] +
               self.observation_space["desired_goal"].high.shape[0],)
      self.observation_space = spaces.Box(-np.inf,
                                          np.inf,
                                          shape=shape,
                                          dtype="float32")

  def _transform_state(self, state):
    """
    modify state to contain: observation, achieved_goal, desired_goal
    """
    if type(state) == dict:
      state = np.concatenate((state["observation"], state["desired_goal"]))
    return state


class EnvManager:

  def __init__(self,
               env_name,
               env_args={},
               r_scale=1,
               r_shift=0.0,
               with_goal=True):
    self.make_env = None
    # Search from our own environments
    if env_name == "Reach2D":
      env_args["sparse"] = True
      self.make_env = lambda: reacher_2d.make("Reacher", **env_args)
    elif env_name == "Reach2DDense":
      env_args["sparse"] = False
      self.make_env = lambda: reacher_2d.make("Reacher", **env_args)
    elif env_name == "Reach2DF":
      env_args["order"] = 1
      env_args["sparse"] = True
      self.make_env = lambda: reacher_2d.make("Reacher", **env_args)
    elif env_name == "Reach2DFDense":
      env_args["order"] = 1
      env_args["sparse"] = False
      self.make_env = lambda: reacher_2d.make("Reacher", **env_args)
    elif env_name == "BlockReachF":
      env_args["sparse"] = True
      env_args["order"] = 1
      env_args["block"] = True
      self.make_env = lambda: reacher_2d.make("Reacher", **env_args)
    elif env_name == "BlockReachFDense":
      env_args["sparse"] = False
      env_args["order"] = 1
      env_args["block"] = True
      self.make_env = lambda: reacher_2d.make("Reacher", **env_args)

    # Search in DMC Envs
    if self.make_env is None and dmc is not None and ":" in env_name:
      # acrobot swingup, acrobot swingup_sparse, ball_in_cup catch,
      # cartpole balance, cartpole balance_sparse, cartpole swingup,
      # cartpole swingup_sparse, cheetah run, finger spin, finger turn_easy,
      # finger turn_hard, fish upright, fish swim, hopper stand, hopper hop,
      # humanoid stand, humanoid walk, humanoid run, manipulator bring_ball,
      # pendulum swingup, point_mass easy, reacher easy, reacher hard, swimmer
      # swimmer6, swimmer swimmer15, walker stand, walker walk, walker run
      domain_name, task_name = env_name.split(":")
      try:
        dmc.make(domain_name=domain_name, task_name=task_name)
        self.make_env = lambda: dmc.make(domain_name=domain_name,
                                         task_name=task_name)
      except ValueError:
        pass

    # OpenAI Gym envs: https://gym.openai.com/envs/#mujoco
    # D4RL envs: https://github.com/rail-berkeley/d4rl/wiki/Tasks
    # Customized envs: YWPickAndPlaceRandInit-v0, YWPegInHoleRandInit-v0
    #                  YWPegInHole2D-v0
    if self.make_env is None and gym is not None:
      try:
        gym.make(env_name, **env_args)
        self.make_env = lambda: gym.make(env_name, **env_args)
      except gym.error.UnregisteredEnv:
        pass

    # Search in MetaWorld envs
    if (self.make_env is None and mtw_envs is not None and
        env_name in mtw_envs.keys()):

      def make_env():
        args = mtw_args[env_name]["args"]
        kwargs = mtw_args[env_name]["kwargs"]
        kwargs["random_init"] = (env_args["random_init"]
                                 if "random_init" in env_args.keys() else False
                                )  # disable random goal locations
        kwargs["obs_type"] = "with_goal"  # disable random goal locations
        kwargs["rewMode"] = "sparse"  # use sparse reward mode by default
        env = mtw_envs[env_name](*args, **kwargs)
        return env

      self.make_env = make_env

    if self.make_env is None:
      raise NotImplementedError

    # Add extra properties on the environment.
    self.r_scale = r_scale
    self.r_shift = r_shift
    self.with_goal = with_goal

  def get_env(self):
    if self.with_goal:
      return GoalEnvWrapper(self.make_env, self.r_scale, self.r_shift)
    return NoGoalEnvWrapper(self.make_env, self.r_scale, self.r_shift)


if __name__ == "__main__":
  env_manager = EnvManager("hopper-medium-replay-v0")
  env = env_manager.get_env()
  dataset = env.get_dataset()
  env.seed(0)
  done = True
  while True:
    if done:
      obs = env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()
