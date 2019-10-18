from td3fd.env import point_reach

try:
    from td3fd.env.franka_env import panda_env
except:
    panda_env = None
try:
    import gym
    from mujoco_py import GlfwContext

    GlfwContext(offscreen=True)  # Create a window to init GLFW.
except:
    gym = None


class EnvWrapper:
    def __init__(self, make_env, r_scale, r_shift, eps_length):
        self.env = make_env()
        self.r_scale = r_scale
        self.r_shift = r_shift
        assert eps_length > 0, "for now, must provide eps_length"
        self.eps_length = eps_length
        # need the following properties
        self.max_u = self.env.max_u if hasattr(self.env, "max_u") else 1  # note that 1 is just for most envs
        self.action_space = self.env.action_space

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)
        reward = (reward + self.r_shift) / self.r_scale
        return reward

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        return state

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def seed(self, seed=0):
        return self.env.seed(seed)

    def step(self, action):
        state, r, extra, info = self.env.step(action)
        r = (r + self.r_shift) / self.r_scale
        return state, r, extra, info

    def close(self):
        return self.env.close()


class EnvManager:
    def __init__(self, env_name, env_args={}, r_scale=1, r_shift=0.0, eps_length=0):
        self.make_env = None
        # Search from our own environments
        if env_name == "Reach2D":
            env_args["dim"] = 2
            env_args["sparse"] = True
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "Reach2DDense":
            env_args["dim"] = 2
            env_args["sparse"] = False
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "Reach2DF":
            env_args["dim"] = 2
            env_args["order"] = 1
            env_args["sparse"] = True
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "Reach2DFDense":
            env_args["dim"] = 2
            env_args["order"] = 1
            env_args["sparse"] = False
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "Reach1D":
            env_args["dim"] = 1
            env_args["sparse"] = True
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "Reach1DDense":
            env_args["dim"] = 1
            env_args["sparse"] = False
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "Reach1DF":
            env_args["dim"] = 1
            env_args["order"] = 1
            env_args["sparse"] = True
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "Reach1DFDense":
            env_args["dim"] = 1
            env_args["order"] = 1
            env_args["sparse"] = False
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "BlockReachF":
            env_args["sparse"] = True
            env_args["order"] = 1
            env_args["block"] = True
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "BlockReachFDense":
            env_args["sparse"] = False
            env_args["order"] = 1
            env_args["block"] = True
            self.make_env = lambda: point_reach.make("Reacher", **env_args)

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
    env_manager = EnvManager("YWFetchPegInHolePixel-v0", eps_length=40)
    env = env_manager.get_env()

    env.seed(0)
    for i in range(1000):
        env.reset()
        action = np.random.randn(env.action_space.shape[0])  # sample random action
        state, r, extra, info = env.step(action)
        print(state.keys(), r, extra, info)
        for k, i in state.items():
            print(i.shape)
        import matplotlib.pyplot as plt

        plt.imshow(state["pixel"])
        plt.show()

        input("Press Enter to continue...")
        env.render()
