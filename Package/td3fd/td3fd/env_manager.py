import numpy as np

from td3fd.env import reacher_2d

try:
    from td3fd.env.franka_env import panda_env
except:
    panda_env = None
try:
    import gym

    # Need this for getting observation in pixels (for vision based learning) (for future works)
    # from mujoco_py import GlfwContext
    # GlfwContext(offscreen=True)  # Create a window to init GLFW.
except:
    gym = None


class EnvWrapper:
    """Wrapper of the environment that does the following:
        1. adjust rewards: r = (r + r_shift) / r_scale
        2. modify state to contain: observation, achieved_goal, desired_goal
    """

    def __init__(self, make_env, r_scale, r_shift, eps_length):
        self.env = make_env()
        self.r_scale = r_scale
        self.r_shift = r_shift
        self.eps_length = eps_length if eps_length else self.env._max_episode_steps
        # need the following properties
        self.max_u = self.env.max_u if hasattr(self.env, "max_u") else 1  # note that 1 is just for most envs
        self.action_space = self.env.action_space

    # def compute_reward(self, achieved_goal, desired_goal, info):
    #     reward = self.env.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)
    #     reward = (reward + self.r_shift) / self.r_scale
    #     return reward

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        return self._transform_state(state)

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
    
    def _transform_state(self, state):
        """
        modify state to contain: observation, achieved_goal, desired_goal
        """
        if not type(state) == dict:
            state = {"observation": state, "achieved_goal": np.empty(0), "desired_goal": np.empty(0)}
        return state


class EnvManager:
    def __init__(self, env_name, env_args={}, r_scale=1, r_shift=0.0, eps_length=0):
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

        # Search from openai gym
        if self.make_env is None and gym is not None:
            _ = gym.make(env_name, **env_args)
            self.make_env = lambda: gym.make(env_name, **env_args)

        # Franka environment
        if self.make_env is None and panda_env is not None:
            # TODO add a make function
            self.make_env = panda_env.FrankaPegInHole

        if self.make_env is None:
            raise NotImplementedError

        # Add extra properties on the environment.
        self.r_scale = r_scale
        self.r_shift = r_shift
        self.eps_length = eps_length

    def get_env(self):
        return EnvWrapper(self.make_env, self.r_scale, self.r_shift, self.eps_length)


if __name__ == "__main__":
    import numpy as np

    # For a openai env
    env_manager = EnvManager("Reacher-v2")
    env = env_manager.get_env()

    env.seed(0)
    done = True
    for i in range(1000):
        if done:
            env.reset()
        action = np.random.randn(env.action_space.shape[0])  # sample random action
        state, r, done, info = env.step(action)
        print(state, r, done, info)
        env.render()
