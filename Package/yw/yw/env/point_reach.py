"""Reacher Environment Implementation
"""
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt

def make(env_name, **env_args):
    # note: we only have one environment
    try:
        _ = Reach2D(**env_args)
    except:
        return NotImplementedError

    return Reach2D(**env_args)

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
        else:  # self.sparse == True
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

    def reset(self, init=None):

        # Randomly select the initial state
        if self.order == 2:

            # self.speed = np.zeros(2)
            self.speed = 2 * (self.random.rand(2) - 0.5) * self.boundary

            # self.goal = np.zeros(2)
            self.goal = 2 * (self.random.rand(2) - 0.5) * self.boundary

            self.curr_pos = 2 * (self.random.rand(2) - 0.5) * self.boundary

        else:  # 1

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
        else:  # 1
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
        else:  # 1
            obs = self.curr_pos
        g = self.goal
        ag = self.curr_pos
        r = self.compute_reward(ag, g)
        # g = np.empty((0))
        # ag = np.empty((0))
        if self.order == 2:
            position_ok = r > -self.threshold if self.sparse == False else r == 1
            # position_ok = r > 0.9 if self.sparse == False else r == 1
            speed_ok = np.sqrt(np.sum(np.square(self.speed))) < 0.25
            is_success = position_ok and speed_ok
        else:
            is_success = r > -self.threshold if self.sparse == False else r == 1
        return ({"observation": obs, "desired_goal": g, "achieved_goal": ag}, r, 0, {"is_success": is_success})


if __name__ == "__main__":
    env = Reach2D()
    env.reset()
    for i in range(32):
        env.step([-np.sin(i / 10), -np.cos(i / 10)])
        env.render()
