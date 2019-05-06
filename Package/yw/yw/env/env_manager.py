from yw.env import point_reach
import gym


class EnvManager:
    def __init__(self, env_name, r_scale=1, r_shift = 0.05, eps_length = 0):
        self.make_env = None
        self.env_arg = {}

        # Search from our own environments
        if env_name == "Reach1D":
            self.make_env = point_reach.Reach1D
        elif env_name == "Reach2D":
            self.make_env = point_reach.Reach2D
        elif env_name == "Reach2DSparse":
            self.make_env = point_reach.Reach2D
            self.env_arg["sparse"] = True
        elif env_name == "Reach2DFirstOrder":
            self.make_env = point_reach.Reach2D
            self.env_arg["order"] = 1
        elif env_name == "FakeData":
            self.make_env = point_reach.FakeData

        # Search from openai gym
        if self.make_env == None:
            try:
                _ = gym.make(env_name)
                self.make_env = lambda: gym.make(env_name)
            except (NotImplementedError):
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
            state, r, extra, is_success = self.env.step(action)
            r = (r + self.r_shift) / self.r_scale
            return state, r, extra, is_success


if __name__ == "__main__":
    env_manager = EnvManager("Reach2D", 1)
    env = env_manager.get_env()

