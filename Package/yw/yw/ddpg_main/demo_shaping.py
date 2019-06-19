import tensorflow as tf


class DemoShaping:
    def __init__(self, o, g, u, o_2, g_2, u_2, gamma):
        """
        Implement the state, action based potential function and corresponding actions
        Args:
            o - observation
            g - goal
            u - action
            o_2 - observation that goes to
            g_2 - same as g, idk why I must make this specific
            u_2 - output from the actor of the main network
        """
        # Calculate potential
        self.potential = self.calc_potential(o, g, u)
        # Calculate reward
        potential = self.calc_potential(o, g, u)
        next_potential = self.calc_potential(o_2, g_2, u_2)
        self.reward = gamma * next_potential - potential

    def calc_potential(self, o, g, u):
        """
        Just return negative value of distance between current state and goal state
        """
        # ret = -tf.abs(o - g) * 10
        ret = -((g - o) * 10 - u) ** 2 / 5
        return ret
