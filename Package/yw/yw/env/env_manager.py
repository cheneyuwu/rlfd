from yw.env import point_reach
import gym
import yw.env.suite_wrapper as suite


class EnvManager:
    def __init__(self, env_name, env_args={}, r_scale=1, r_shift=0.05, eps_length=0):
        self.make_env = None
        self.env_arg = {}
        # Search from our own environments
        if env_name == "Reach2D":
            self.make_env = point_reach.Reach2D
        elif env_name == "Reach2DSparse":
            self.make_env = point_reach.Reach2D
            self.env_arg["sparse"] = True
        elif env_name == "Reach2DFirstOrder":
            self.make_env = point_reach.Reach2D
            self.env_arg["order"] = 1

        # Search from openai gym
        if self.make_env == None:
            try:
                _ = gym.make(env_name, **env_args)
                self.make_env = lambda: gym.make(env_name, **env_args)
            except:
                pass

        if self.make_env == None:
            try:
                # there's no easy way to pass this
                env_args = {
                    "has_renderer": False,  # no on-screen renderer
                    "has_offscreen_renderer": False,  # no off-screen renderer
                    "use_object_obs": True,  # use object-centric feature
                    "use_camera_obs": False,  # no camera observations)
                }
                _ = suite.make(env_name, env_args)
                self.make_env = lambda: suite.make(env_name, env_args)
            except:
                pass

        if self.make_env is None:
            raise NotImplementedError

        # Add extra properties on the environment.
        self.r_scale = r_scale
        self.r_shift = r_shift
        self.eps_length = eps_length

    def get_env(self):
        return EnvManager.EnvWrapper(self.make_env, self.env_arg, self.r_scale, self.r_shift, self.eps_length)

    class EnvWrapper:
        def __init__(self, make_env, env_arg, r_scale, r_shift, eps_length):
            self.env = make_env(**env_arg)
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


if __name__ == "__main__":
    import numpy as np

    # For a robosuite env
    env_manager = EnvManager("SawyerLift")
    env = env_manager.get_env()
    env.reset()
    env.seed(0)
    for i in range(3):
        action = np.random.randn(env.action_space.shape[0])  # sample random action
        state, r, extra, info = env.step(action)
        print(state)
        print(r)
        print(extra)
        print(info)
        # env.render()

    # For a openai env
    env_manager = EnvManager("FetchPickAndPlace-v1")
    env = env_manager.get_env()
    env.reset()
    env.seed(0)
    for i in range(3):
        action = np.random.randn(env.action_space.shape[0])  # sample random action
        state, r, extra, info = env.step(action)
        print(state)
        print(r)
        print(extra)
        print(info)
        # env.render()

