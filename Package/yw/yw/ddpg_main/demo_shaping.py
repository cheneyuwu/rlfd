import tensorflow as tf

# class DemoShaping:
#     def __init__(self, gamma):
#         self.gamma = gamma
    
#     def get_reward(self, o, g, u, o_2, g_2, u_2, **kwargs):
#         """
#         Compute the reward based on discounted difference of potentials.
#         """
#         return self.gamma * self.get_potential(o_2, g_2, u_2) - self.get_potential(o, g, u)
    
#     def get_potential(self, o, g, u):
#         pass



# class ReacherStateDemoShaping(DemoShaping):
#     def __init__(self, gamma):
#         super().__init__(gamma)
    
#     def get_potential(self, o, g, u):
#         """
#         Just return negative value of distance between current state and goal state
#         """
#         return -np.abs(o-g)


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
            next_potential = self.calc_potential(o_2, g_2, u_2[i])
            self.reward.append(10 * (gamma * next_potential - self.potential))
    
    def calc_potential(self, o, g, u):
        """
        Just return negative value of distance between current state and goal state
        """
        ret = -tf.abs(o-g+0.5)
        return ret