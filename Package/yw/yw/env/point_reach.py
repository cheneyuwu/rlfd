"""
Environment Requirement:
    Need goal, observation, action -> currently we consider them as continuous
    Implement the following methods: step, reset, render
"""
import math
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt


class FakeData:
    """ This will return an environment with all the required methods. Input dimensions are all (1,).
        Use this for debugging only!
    """

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


class Reach1D:
    def __init__(self, seed=0):
        self.random = np.random.RandomState(seed)
        self.interval = 0.1
        self.mass = 1
        self._max_episode_steps = 32
        self.max_u = 5
        self.action_space = self.ActionSpace()
        plt.ion()
        plt.show()
        self.reset()

    class ActionSpace:
        def __init__(self, seed=0):
            self.random = np.random.RandomState(seed)
            self.shape = np.zeros(1).shape

        def sample(self):
            return self.random.rand(1)

    def compute_reward(self, achieved_goal, desired_goal, info=0):
        achieved_goal.reshape(-1, 1)
        desired_goal.reshape(-1, 1)
        return -abs(achieved_goal - desired_goal)

    def render(self):
        plt.subplot(211)
        plt.cla()
        plt.axis([0, 50, -1, 1])
        plt.plot(self.history["t"], self.history["position"], "r")
        plt.plot(self.history["t"], self.history["goal"], "g")
        plt.subplot(212)
        plt.plot(self.history["t"], self.history["r"], "g")
        plt.show()
        plt.pause(0.05)

    def reset(self):
        self.speed = np.zeros(1)
        self.goal = self.random.rand(1) - 0.5
        self.curr_pos = self.random.rand(1) - 0.5
        self.T = 0
        self.history = {"position": [], "goal": [], "t": [], "r": []}
        plt.clf()
        return self._get_state()[0]

    def seed(self, seed=0):
        self.random = np.random.RandomState(seed)

    def step(self, action):
        action = max(min(action * 10, self.max_u), -self.max_u)
        acc = np.array((action / self.mass))
        self.curr_pos = self.curr_pos + self.speed * self.interval + self.interval * self.interval * acc * 0.5
        self.speed = self.speed + acc * self.interval
        self.T = self.T + 1
        self.history["position"].append(self.curr_pos)
        self.history["goal"].append(self.goal)
        self.history["t"].append(self.T)
        _, r, _, _ = self._get_state()
        self.history["r"].append(r)
        return self._get_state()

    def _get_state(self):
        obs = np.concatenate((self.speed, self.curr_pos))
        g = self.goal
        ag = self.curr_pos
        r = sum(-abs(ag - g))
        return ({"observation": obs, "desired_goal": g, "achieved_goal": ag}, r, 0, {"is_success": (r > -0.02)})

class Reach2D:
    def __init__(self, order=2, sparse=False, seed=0):
        self.random = np.random.RandomState(seed)
        self.order = order
        self.sparse = sparse
        self.interval = 0.1
        self.mass = 1
        self.boundary = 1.0
        self.threshold = self.boundary / 12
        self._max_episode_steps = 42 if self.order == 2 else 24
        self.max_u = 2
        self.action_space = self.ActionSpace()
        plt.ion()
        plt.show()
        self.reset()

    class ActionSpace:
        def __init__(self, seed=0):
            self.random = np.random.RandomState(seed)
            self.shape = np.zeros(2).shape

        def sample(self):
            return self.random.rand(2)

    def compute_reward(self, achieved_goal, desired_goal, info=0):
        achieved_goal = achieved_goal.reshape(-1, 2)
        desired_goal = desired_goal.reshape(-1, 2)
        distance = np.sqrt(np.sum(np.square(achieved_goal - desired_goal), axis=1))
        if self.sparse == False:
            return -distance
            # return 1.0 / (1.0 + distance)
        else: # self.sparse == True
            return (distance < self.threshold).astype(np.int64)

    def render(self):
        plt.clf()
        plt.subplot(211)
        plt.axis([-self.boundary, self.boundary, -self.boundary, self.boundary])
        plt.plot(self.history["position"][-1][0], self.history["position"][-1][1], "o", color="r")
        plt.plot(self.history["goal"][-1][0], self.history["goal"][-1][1], "o", color="g")
        plt.subplot(212)
        plt.axis([0, self._max_episode_steps, -2, 2])
        plt.plot(self.history["t"], self.history["r"], "g")
        plt.plot(self.history["t"], self.history["v"], "r")
        plt.show()
        plt.pause(0.05)

    def reset(self, init = None):

        # Randomly select the initial state
        if self.order == 2:

            # self.speed = np.zeros(2)
            self.speed = 2 * (self.random.rand(2) - 0.5) * self.boundary

            # self.goal = np.zeros(2)
            self.goal = 2 * (self.random.rand(2) - 0.5) * self.boundary

            self.curr_pos = 2 * (self.random.rand(2) - 0.5) * self.boundary

        else: # 1

            # self.goal = np.zeros(2)
            self.goal = 2 * (self.random.rand(2) - 0.5) * self.boundary

            self.curr_pos = 2 * (self.random.rand(2) - 0.5) * self.boundary

        # Make it possible to override the initial state.
        if init is not None:
            if "observation" in init.keys():
                if self.order == 2:
                    self.curr_pos = init["observation"][2:4]
                else:
                    self.curr_pos = init["observation"][0:2]
            if "goal" in init.keys():
                self.goal = init["goal"][0:2]

        self.T = 0
        self.history = {"position": [], "goal": [], "t": [], "r": [], "v": []}
        return self._get_state()[0]

    def seed(self, seed=0):
        self.random = np.random.RandomState(seed)

    def step(self, action):
        action = np.array(action)
        action.clip(-self.max_u, self.max_u)
        if self.order == 2:
            acc = action / self.mass
            self.curr_pos = self.curr_pos + self.speed * self.interval + self.interval * self.interval * acc * 0.5
            self.speed = self.speed + acc * self.interval
        else: # 1
            self.curr_pos = self.curr_pos + action * self.interval
        self.T = self.T + 1
        self.history["position"].append(self.curr_pos)
        self.history["goal"].append(self.goal)
        self.history["t"].append(self.T)
        _, r, _, info = self._get_state()
        self.history["r"].append(r)
        self.history["v"].append(info["is_success"])
        return self._get_state()

    def _get_state(self):
        if self.order == 2:
            obs = np.concatenate((self.speed, self.curr_pos))
        else: # 1
            obs = self.curr_pos
        g = self.goal
        ag = self.curr_pos
        r = self.compute_reward(ag, g)
        if self.order == 2:
            position_ok = r > -self.threshold if self.sparse == False else r == 1
            # position_ok = r > 0.9 if self.sparse == False else r == 1
            speed_ok = np.sqrt(np.sum(np.square(self.speed))) < 0.25
            is_success = position_ok and speed_ok
        else:
            is_success = r > -self.threshold if self.sparse == False else r == 1
        return (
            {"observation": obs, "desired_goal": g, "achieved_goal": ag},
            r,
            0,
            {"is_success": is_success},
        )

if __name__ == "__main__":
    a = Reach2D()
    while True:
        a.reset()
        for i in range(100):
            k, _, r, info = a.step([-math.sin(i / 10), -math.cos(i / 10)])
            a.render()
