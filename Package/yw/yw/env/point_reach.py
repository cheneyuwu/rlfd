"""Reacher Environment Implementation
"""
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  #  TkAgg Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def make(env_name, **env_args):
    # note: we only have one environment
    try:
        _ = eval(env_name)(**env_args)
    except:
        raise NotImplementedError
    return eval(env_name)(**env_args)


class Reacher:
    """
    2D reacher environment with blocks
    """

    def __init__(self, dim=2, order=2, sparse=False, block=False, seed=0):
        self.random = np.random.RandomState(seed)
        self.order = order
        self.sparse = sparse
        self.dim = dim
        self.interval = 0.1  # use 0.04 for block reach
        self.mass = 1
        self.boundary = 1.0
        self.threshold = self.boundary / 12
        self._max_episode_steps = 42 if self.order == 2 else 20
        self.max_u = 2
        self.action_space = self.ActionSpace(self.dim)
        if self.dim == 2:  # create a workspace boundary only when dim is 2
            self.workspace = self.Block((-self.boundary, -self.boundary), 2 * self.boundary, 2 * self.boundary)
        self.blocks = []
        if block == True:
            assert self.dim == 2, "cannot have block for dimensions other than 2!"
            # add blocks manually added
            self.blocks.append(self.Block((-0.5, -0.5), 1.0, 1.0))
        plt.ion()
        # plt.show()
        self.reset()

    class ActionSpace:
        def __init__(self, dim, seed=0):
            self.dim = dim
            self.random = np.random.RandomState(seed)
            self.shape = np.zeros(self.dim).shape

        def sample(self):
            return self.random.rand(self.dim)

    class Block:
        def __init__(self, start, width, height):
            """
            start (float) - (x, y) as the center of the square
            height
            width
            """
            self.start = start
            self.height = height
            self.width = width

        def plot(self, ax):
            rect = patches.Rectangle(self.start, self.width, self.height, linewidth=1, edgecolor="r", facecolor="none")
            # Add the patch to the Axes
            ax.add_patch(rect)

        def inside(self, point):
            return (
                point[0] > self.start[0]
                and point[0] < self.start[0] + self.width
                and point[1] > self.start[1]
                and point[1] < self.start[1] + self.height
            )

        def outside(self, point):
            return (
                point[0] <= self.start[0]
                or point[0] >= self.start[0] + self.width
                or point[1] <= self.start[1]
                or point[1] >= self.start[1] + self.height
            )

    def compute_reward(self, achieved_goal, desired_goal, info=0):
        distance = self._compute_distance(achieved_goal, desired_goal)
        if self.sparse == False:
            return -distance
            # return np.maximum(-0.5, -distance)
            # return 0.2 / (0.2 + distance)
        else:  # self.sparse == True
            return (distance < self.threshold).astype(np.int64)

    def _compute_distance(self, achieved_goal, desired_goal):
        achieved_goal = achieved_goal.reshape(-1, self.dim)
        desired_goal = desired_goal.reshape(-1, self.dim)
        distance = np.sqrt(np.sum(np.square(achieved_goal - desired_goal), axis=1))
        return distance

    def render(self):

        plt.clf()
        # plot the environment visualization
        ax = plt.subplot(211)
        if self.dim == 1:
            ax.axis([0, self._max_episode_steps, -2, 2])
            ax.plot(self.history["t"], self.history["position"], color="r", label="current position")
            ax.plot(self.history["t"], self.history["goal"], color="g", label="goal position")
            ax.set_xlabel("time")
            ax.set_ylabel("position")
        elif self.dim == 2:
            ax.axis([-self.boundary, self.boundary, -self.boundary, self.boundary])
            ax.plot(
                self.history["position"][-1][0],
                self.history["position"][-1][1],
                "o",
                color="r",
                label="current position",
            )
            ax.plot(self.history["goal"][-1][0], self.history["goal"][-1][1], "o", color="g", label="goal position")
            for block in self.blocks:
                block.plot(ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            assert False, "Cannot render when number of dimension is greate than 2."
        ax.set_title("state")
        ax.legend(loc="upper right")
        # plot reward and success info
        ax = plt.subplot(212)
        ax.axis([0, self._max_episode_steps, -2, 2])
        ax.plot(self.history["t"], self.history["r"], color="g", label="reward")
        ax.plot(self.history["t"], self.history["v"], color="r", label="is_success")
        ax.legend()
        ax.set_title("info")

        plt.show()
        plt.pause(0.05)

    def reset(self):

        # The initial state and final goal is fixed
        self.goal = self.random.uniform(-0.0, 0.0) * np.ones(self.dim) * self.boundary
        self.curr_pos = self.random.uniform(-0.8, -0.8) * np.ones(self.dim) * self.boundary
        if self.order == 2:
            self.speed = np.zeros(self.dim)

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
            new_curr_pos = self.curr_pos + self.speed * self.interval + self.interval * self.interval * acc * 0.5
            new_speed = self.speed + acc * self.interval
            if self.dim != 2 or (
                all([block.outside(new_curr_pos) for block in self.blocks]) and self.workspace.inside(new_curr_pos)
            ):
                self.curr_pos = new_curr_pos
                self.speed = new_speed
            else:
                self.speed = 0.0 * self.speed
        else:  # 1
            new_curr_pos = self.curr_pos + action * self.interval
            if self.dim != 2 or (
                all([block.outside(new_curr_pos) for block in self.blocks]) and self.workspace.inside(new_curr_pos)
            ):
                self.curr_pos = new_curr_pos

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
        # return distance as metric to measure performance
        distance = self._compute_distance(ag, g)
        # is_success or not
        is_success = distance < self.threshold
        if self.order == 2:
            is_success = is_success and np.sqrt(np.sum(np.square(self.speed))) < 0.25
        return (
            {"observation": obs, "desired_goal": g, "achieved_goal": ag},
            r,
            0,
            {"is_success": is_success, "shaping_reward": -distance},
        )


if __name__ == "__main__":

    env = make("Reacher", dim=1, order=1)
    obs = env.reset()
    for i in range(32):
        action = obs["desired_goal"] - obs["observation"]
        obs, r, done, info = env.step(action)
        env.render()

    env = make("Reacher", dim=2)
    obs = env.reset()
    for i in range(32):
        obs, r, done, info = env.step([-np.sin(i / 10), -np.cos(i / 10)])
        env.render()

    env = make("Reacher", dim=2, order=1, block=True)
    obs = env.reset()
    for i in range(32):
        action = obs["desired_goal"] - obs["observation"]
        obs, r, done, info = env.step(action)
        env.render()
    for i in range(32):
        action = -obs["desired_goal"] + obs["observation"]
        obs, r, done, info = env.step(action)
        env.render()
