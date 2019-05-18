"""This will return an environment with all the required methods. Input dimensions are all (1,).
    Use this for debugging only!
"""

import numpy as np


class FakeData:
    def __init__(self, seed=0):
        self.random = np.random.RandomState(seed)
        self._max_episode_steps = 1
        self.max_u = 1
        self.action_space = self.ActionSpace()

    class ActionSpace:
        def __init__(self, seed=0):
            self.random = np.random.RandomState(seed)
            self.shape = np.zeros(1).shape

        def sample(self):
            return self.random.rand(1)

    def compute_reward(self, a, b, c=0):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def reset(self):
        return 0

    def seed(self):
        raise NotImplementedError

    def step(self, action):
        return (
            {"observation": np.zeros((1,)), "desired_goal": np.zeros((1,)), "achieved_goal": np.zeros((1,))},
            0,
            0,
            {"is_success": 0},
        )

    def _get_state(self):
        raise NotImplementedError


if __name__ == "__main__":
    env = FakeData()
    env.reset()
    for i in range(32):
        env.step([-np.sin(i / 10), -np.cos(i / 10)])
