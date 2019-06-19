import numpy as np

class DemoShaping:
    def __init__(self, gamma):
        self.gamma = gamma
    
    def get_reward(self, o, g, u, o_2, g_2, u_2, **kwargs):
        """
        Compute the reward based on discounted difference of potentials.
        """
        return self.gamma * self.get_potential(o_2, g_2, u_2) - self.get_potential(o, g, u)
    
    def get_potential(self, o, g, u):
        pass



class ReacherStateDemoShaping(DemoShaping):
    def __init__(self, gamma):
        super().__init__(gamma)
    
    def get_potential(self, o, g, u):
        """
        Just return negative value of distance between current state and goal state
        """
        return -np.abs(o-g)