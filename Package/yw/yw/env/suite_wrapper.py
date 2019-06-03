import numpy as np

import robosuite


def make(env_name, env_args):
    try:
        _ = robosuite.make(env_name, **env_args)
    except:
        return NotImplementedError

    make_env = lambda: robosuite.make(env_name, **env_args)
    return SuiteWrapper(make_env)


class SuiteWrapper:
    def __init__(self, make_env):
        self.env = make_env()
        # need the following properties
        self._max_episode_steps = 64  # reset this later
        action_max = np.concatenate((-self.env.action_spec[0], self.env.action_spec[1]))
        self.max_u = np.max(action_max) # note that 1 is just for most envs
        assert all([v == self.max_u for v in action_max])
        assert type(self.env.dof) == int
        self.action_space = self.ActionSpace(self.env.dof)

    class ActionSpace:
        def __init__(self, env_dof, seed=0):
            self.env_dof = env_dof
            self.random = np.random.RandomState(seed)
            self.shape = np.zeros(self.env_dof).shape

        def sample(self):
            return self.random.rand(self.env_dof)

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        state = {
            "observation": np.concatenate((state["robot-state"], state["object-state"])),
            "desired_goal": np.empty(0),
            "achieved_goal": np.empty(0),
        }
        return state

    def render(self):
        return self.env.render()

    def seed(self, seed=0):
        pass
        # return self.env.seed(seed)

    def step(self, action):
        state, r, extra, info = self.env.step(action)
        state = {
            "observation": np.concatenate((state["robot-state"], state["object-state"])),
            "desired_goal": np.empty(0),
            "achieved_goal": np.empty(0),
        }
        info["is_success"] = float(self.env._check_success())
        return state, r, extra, info
