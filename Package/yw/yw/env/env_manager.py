from yw.env import point_reach
import gym
import yw.env.suite_wrapper as suite


class EnvManager:
    def __init__(self, env_name, env_args={}, r_scale=1, r_shift=0.05, eps_length=0):
        self.make_env = None
        # Search from our own environments
        if env_name == "Reach2D":
            env_args["dim"] = 2
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "Reach2DSparse":
            env_args["dim"] = 2
            env_args["sparse"] = True
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "Reach2DFirstOrderSparse":
            env_args["dim"] = 2
            env_args["order"] = 1
            env_args["sparse"] = True
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "Reach2DFirstOrder":
            env_args["dim"] = 2
            env_args["order"] = 1
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "Reach1D":
            env_args["dim"] = 1
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "Reach1DFirstOrder":
            env_args["dim"] = 1
            env_args["order"] = 1
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "Reach1DFirstOrderSparse":
            env_args["dim"] = 1
            env_args["order"] = 1
            env_args["sparse"] = True
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "Reach1DSparse":
            env_args["dim"] = 1
            env_args["sparse"] = True
            self.make_env = lambda: point_reach.make("Reacher", **env_args)
        elif env_name == "BlockReachFirstOrder":
            env_args["sparse"] = False
            env_args["order"] = 1
            self.make_env = lambda: point_reach.make("BlockReacher", **env_args)
        elif env_name == "BlockReachFirstOrderSparse":
            env_args["sparse"] = True
            env_args["order"] = 1
            self.make_env = lambda: point_reach.make("BlockReacher", **env_args)

        # Search from openai gym
        if self.make_env == None:
            try:
                _ = gym.make(env_name, **env_args)
                self.make_env = lambda: gym.make(env_name, **env_args)
            except:
                pass

        if self.make_env == None:
            try:
                tmp = suite.make(env_name, **env_args)
                tmp.close()
                self.make_env = lambda: suite.make(env_name, **env_args)
            except:
                pass

        if self.make_env is None:
            raise NotImplementedError

        # Add extra properties on the environment.
        self.r_scale = r_scale
        self.r_shift = r_shift
        self.eps_length = eps_length

    def get_env(self):
        return EnvManager.EnvWrapper(self.make_env, self.r_scale, self.r_shift, self.eps_length)

    class EnvWrapper:
        def __init__(self, make_env, r_scale, r_shift, eps_length):
            self.env = make_env()
            self.r_scale = r_scale
            self.r_shift = r_shift
            self.eps_length = eps_length
            # need the following properties
            self._max_episode_steps = self.eps_length if self.eps_length else self.env._max_episode_steps
            self.max_u = self.env.max_u if hasattr(self.env, "max_u") else 1  # note that 1 is just for most envs
            self.action_space = self.env.action_space

        def compute_reward(self, achieved_goal, desired_goal, info):
            reward = self.env.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)
            reward = (reward + self.r_shift) / self.r_scale
            return reward

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def render(self):
            return self.env.render()

        def seed(self, seed=0):
            return self.env.seed(seed)

        def step(self, action):
            state, r, extra, info = self.env.step(action)
            r = (r + self.r_shift) / self.r_scale
            return state, r, extra, info

        def close(self):
            return self.env.close()



if __name__ == "__main__":
    import numpy as np

    # For a robosuite env
    # env_manager = EnvManager(
    #     "SawyerLift", env_args={"has_renderer": True, "ignore_done": True, "use_camera_obs": False, "control_freq": 50}
    # )
    # env = env_manager.get_env()
    # env.reset()
    # for i in range(100):
    #     action = np.random.randn(env.action_space.shape[0])  # sample random action
    #     state, r, extra, info = env.step(action)
    #     env.render()

    # For a openai env
    env_manager = EnvManager("FetchReach-v1")
    env = env_manager.get_env()
    
    env.seed(0)
    for i in range(1000):
        env.reset()
        # action = np.random.randn(env.action_space.shape[0])  # sample random action
        # state, r, extra, info = env.step(action)
        input("Press Enter to continue...")
        # print(state, r, extra, info)
        env.render()