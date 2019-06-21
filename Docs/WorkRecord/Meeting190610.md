# RL + IL Research Project
Meeting Summary\
June 10th, 2019


## Tasks
1. Combine IL and RL by training DDPG critic using demonstration data.
    - Demonstration trajectories $(s_1, a_1, r_1, s_2, a_2, r_2, ..., s_{n-1}, a_{n_1}, r_{n-1}, s_n)$.
    - Using the demo trajectories to get $(s_t, a_t, q_t)$ where $q_t=r_t + \gamma r_{t+1} + ... + \gamma^{n-1-t} r_{n-1}$
    - Add these transition tuples with the corresponding cumulative reward ($q_t$) to a demonstration buffer. (The demonstration buffer should contain tuples of $(s_t, a_t, q_t, s_{t+1})$.)
    - Update the loss function for training the DDPG critic to be: $\sum_{(s,a,r,s')}(Q_{rl} - Y)^2 + \lambda*\sum_{(s,a,q,s')}(Q_{rl} - \hat{Q})^2$\
    The first term is the bellman error, while the second term is a MSE loss w.r.t. demonstration data. $\hat{Q}$ is the expected cumulative reward calculated based on the demonstration trajectory.
    - With the above change to DDPG, we can explore the following ideas:
        1. Actively select the value of $\lambda$ based on some critics so that we can choose to learn more or less from demonstration.
        2. Consider the problem where the demonstrations can be noisy for some states.
    - However, notice the following:
        1. Do not learn ungrounded Q value from demonstration. Address this problem by using a n-step return instead of using $\hat{Q}$. -> Current implementation uses this.
        2. You should also use L2 regularization to prevent from overfitting to the small dataset
        3. How to make sure that the Q function prefers the action from demonstration after pre-training?
            - Maybe adding a BC loss to actor?
2. More environments/experiments to show that our current method is beneficial.
    - Added robosuite envs, TODO: customized environments?
    - Maybe add the navigation problem, e.g. the  OpenAI CarRacing environment?
3. How to acquire the training set of the demonstration neural net? Or, how to acquire or ask for a demonstration?
    1. Random:
        1. Before training the RL agent, select some states "s", and ask the expert for the expected corresponding action "a" for these states as well as the expected "Q".
        2. Before training the RL agent, select some states and action pairs (s, a), and ask the expert for the expected "Q" for these state action pairs.
    2. Active: (Train RL and Demonstration NN alternatively)
        1. During the RL agent training, based the uncertainty of the output from the critic function in DDPG and the uncertainty from the demonstration neural net, periodically select some states and action pairs (s, a) and ask the expert for the expected "Q" for these state action pairs.


## Issues
1. Convergence rate measured based on success rate.
    - Using a biased demonstration hurts when positive reward.
    - It may also hurt when negative structure
    
2. Uncertainty is affected by:
    1. Hypothesis
        1. Initialization of the network?
            - check negative reward demo training
            - Xavier initialization gives very low initial variance
            - Other initialization does not converge even on this simple problem (tried uniform and normal)
        2. Output of the network?
        3. High variance where no demonstration data after demonstration training.
        4. Whether or not to learn from demonstration during RL training?
    2. Experiments
        1. Negative reward
            1. Random demonstration
                - After demonstration training, high variance at state away from the goal state
                - After RL training, low variance almost everywhere. Slightly higher variance at boundary.
            2. Biased demonstration
                - After demonstration training, high variance at locations where no demonstration
                - After RL training, low variance almost everywhere. Slightly higher variance at boundary.
        2. Positive reward
            1. Random demonstration
                - After demonstration training, high variance (0.2-0.3) near goal state and far from goal state
                - After RL training, the general variance is reduced by ~10 times, variance is noisy, almost low everywhere. (When there is no demonstration, the variance is also... almost uniform everywhere?)
            2. Biased demonstration
                - After dmeonstration traing, very high variance at locations where no demonstration
    3. Some observations
        1. High variance after training on demonstration; Significantly lower variance after trainig on rl.
        2. After demonstration, high variance on states not presenting in demonstration training set;
        3. After rl training, high variance on states not close to the goal state.
        4. High variance at large network output

   
## Meeting Notes


## Reminder