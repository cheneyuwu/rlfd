"""Reacher Environment Implementation
"""
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt


def make(env_name, **env_args):
    # note: we only have one environment
    try:
        _ = Reacher(**env_args)
    except:
        return NotImplementedError

    return Reacher(**env_args)


class Reacher:
    def __init__(self, dim, order=2, sparse=False, seed=0):
        self.random = np.random.RandomState(seed)
        self.order = order
        self.sparse = sparse
        self.dim = dim
        self.interval = 0.1
        self.mass = 1
        self.boundary = 1.0
        self.threshold = self.boundary / 12
        self._max_episode_steps = 42 if self.order == 2 else 16
        self.max_u = 2
        self.action_space = self.ActionSpace(self.dim)
        plt.ion()
        plt.show()
        self.reset()

    class ActionSpace:
        def __init__(self, dim, seed=0):
            self.dim = dim
            self.random = np.random.RandomState(seed)
            self.shape = np.zeros(self.dim).shape

        def sample(self):
            return self.random.rand(self.dim)

    def compute_reward(self, achieved_goal, desired_goal, info=0):
        achieved_goal = achieved_goal.reshape(-1, self.dim)
        desired_goal = desired_goal.reshape(-1, self.dim)
        distance = np.sqrt(np.sum(np.square(achieved_goal - desired_goal), axis=1))
        if self.sparse == False:
            return -distance
            # return 0.5 / (0.5 + distance)
        else:  # self.sparse == True
            return (distance < self.threshold).astype(np.int64)

    def render(self):

        plt.clf()
        plt.subplot(211)
        if self.dim == 1:
            plt.axis([0, self._max_episode_steps, -2, 2])
            plt.plot(self.history["t"], self.history["position"], color="r", label="current position")
            plt.plot(self.history["t"], self.history["goal"], color="g", label="goal position")
            plt.xlabel("time")
            plt.ylabel("position")
        elif self.dim == 2:
            plt.axis([-self.boundary, self.boundary, -self.boundary, self.boundary])
            plt.plot(
                self.history["position"][-1][0],
                self.history["position"][-1][1],
                "o",
                color="r",
                label="current position",
            )
            plt.plot(self.history["goal"][-1][0], self.history["goal"][-1][1], "o", color="g", label="goal position")
            plt.xlabel("x")
            plt.ylabel("y")
        else:
            assert False, "Cannot render when number of dimension is greate than 2."
        plt.title("state")
        plt.legend(loc="upper right")
        plt.subplot(212)
        plt.axis([0, self._max_episode_steps, -2, 2])
        plt.plot(self.history["t"], self.history["r"], color="g", label="reward")
        plt.plot(self.history["t"], self.history["v"], color="r", label="is_success")
        plt.legend()
        plt.title("info")
        plt.show()
        plt.pause(0.05)

    def reset(self, init=None):

        # Randomly select the initial state
        if self.order == 2:
            # self.speed = np.zeros(self.dim)
            self.speed = 2 * (self.random.rand(self.dim) - 0.5) * self.boundary
            # self.goal = np.zeros(self.dim)
            self.goal = 2 * (self.random.rand(self.dim) - 0.5) * self.boundary
            self.curr_pos = 2 * (self.random.rand(self.dim) - 0.5) * self.boundary
        else:  # 1
            self.goal = 0.0 * np.ones(self.dim)
            # self.goal = 2 * (self.random.rand(self.dim) - 0.5) * self.boundary
            # self.curr_pos = -0.7 * np.ones(self.dim) + 0.2 * (self.random.rand(self.dim) - 0.5)
            self.curr_pos = 2 * (self.random.rand(self.dim) - 0.5) * self.boundary

        # Make it possible to override the initial state.
        if init is not None:
            if "observation" in init.keys():
                if self.order == 2:
                    self.curr_pos = init["observation"][self.dim : 2 * self.dim]
                else:
                    self.curr_pos = init["observation"][0 : self.dim]
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
        if self.order == 2:
            position_ok = r > -self.threshold if self.sparse == False else r == 1
            speed_ok = np.sqrt(np.sum(np.square(self.speed))) < 0.25
            is_success = position_ok and speed_ok
        else:
            is_success = r > -self.threshold if self.sparse == False else r == 1
        return ({"observation": obs, "desired_goal": g, "achieved_goal": ag}, r, 0, {"is_success": is_success})


if __name__ == "__main__":

    env = make("", dim=1)
    obs = env.reset()
    for i in range(32):
        obs, r, done, info = env.step([-np.sin(i / 10)])
        env.render()

    env = make("", dim=2)
    obs = env.reset()
    for i in range(32):
        obs, r, done, info = env.step([-np.sin(i / 10), -np.cos(i / 10)])
        env.render()
