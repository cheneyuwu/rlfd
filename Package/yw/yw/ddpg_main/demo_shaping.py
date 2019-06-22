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
        raise NotImplementedError


class ManualDemoShaping(DemoShaping):
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
        super().__init__(o, g, u, o_2, g_2, u_2, gamma)

    def calc_potential(self, o, g, u):
        """
        Just return negative value of distance between current state and goal state
        """
        # ret = -tf.abs(o - g) * 10
        ret = -(tf.clip_by_value((g - o) * 10, -2, 2) - u) ** 2
        return ret


class GaussianDemoShaping(DemoShaping):
    def __init__(self, o, g, u, o_2, g_2, u_2, gamma, demo_inputs_tf):
        """
        Implement the state, action based potential function and corresponding actions
        Args:
            o - observation
            g - goal
            u - action
            o_2 - observation that goes to
            g_2 - same as g, idk why I must make this specific
            u_2 - output from the actor of the main network
            demo_inputs_tf - demo_inputs that contains all the transitons from demonstration
        """
        self.demo_inputs_tf = demo_inputs_tf
        self.sigma = 1 # a hyperparam to be tuned
        self.scale = 10 # another hyperparam to be tuned
        super().__init__(o, g, u, o_2, g_2, u_2, gamma)

    def calc_potential(self, o, g, u):
        """
        Just return negative value of distance between current state and goal state
        """
        # similar to use of sigma
        goal_importance = 1.0
        state_importance = 1.0
        # Concat demonstration inputs
        demo_state_tf = self.demo_inputs_tf["o"] * state_importance
        if self.demo_inputs_tf["g"] != None:
            # for multigoal environments, we have goal as another states
            demo_state_tf = tf.concat(axis=1, values=[demo_state_tf, goal_importance * self.demo_inputs_tf["g"]])
        # demo_state_tf = tf.concat(axis=1, values=[demo_state_tf, self.demo_inputs_tf["u"]])
        # note: shape of demo_state_tf is (num_demo, k), where k is sum of dim o g u

        # Concatenate obs goal and action
        state_tf = o * state_importance
        if g != None:
            # for multigoal environments, we have goal as another states
            state_tf = tf.concat(axis=1, values=[state_tf, goal_importance * g])
        # state_tf = tf.concat(axis=1, values=[state_tf, u])
        # note: shape of state_tf is (batch_size, k), where k is sum of dim o g u

        # Calculate the potential
        # expand dimension of demo_state and state so that they have the same shape: (batch_size, num_demo, k)
        expanded_demo_state_tf = tf.tile(tf.expand_dims(demo_state_tf, 0), [tf.shape(state_tf)[0], 1, 1])
        expanded_state_tf = tf.tile(tf.expand_dims(state_tf, 1), [1, tf.shape(demo_state_tf)[0], 1])
        # calculate distance, result shape is (batch_size, num_demo, k)
        distance_tf = expanded_state_tf - expanded_demo_state_tf
        # calculate L2 Norm square, result shape is (batch_size, num_demo)
        norm_tf = tf.norm(distance_tf, ord=2, axis=-1)
        # cauculate multi var gaussian, result shape is (batch_size, num_demo)
        gaussian_tf = tf.exp(-0.5 * self.sigma * norm_tf * norm_tf) # let sigma be 5
        # sum the result from all demo transitions and get the final result, shape is (batch_size, 1)
        potential = self.scale * tf.reduce_mean(gaussian_tf, axis=-1, keepdims=True)

        return potential
