import tensorflow as tf


class DemoShaping:
    def __init__(self, o, g, u, o_2, g_2, u_2, num_sample, gamma):
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
        self.potential = self.calc_potential(o, g, u)
        self.reward = []
        for i in range(num_sample):
            potential = self.calc_potential(o, g, u)
            next_potential = self.calc_potential(o_2, g_2, u_2[i])
            self.reward.append(10 * (gamma * next_potential - potential))

    def calc_potential(self, o, g, u):
        """
        Just return negative value of distance between current state and goal state
        """
        ret = -tf.abs(o - g)
        return ret
