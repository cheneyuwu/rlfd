# Undergrad Research Project
Meeting Summary<br>
May 27th, 2019

## Goal/Thoughts/Idea from last week

1. Replanning: Combine IL and RL by training DDPG critic using demonstration data.

    - Demonstration trajectories $(s_1, a_1, r_1, s_2, a_2, r_2, ..., s_{n-1}, a_{n_1}, r_{n-1}, s_n)$.
    - Using the demo trajectories to get $(s_t, a_t, q_t)$ where $q_t=r_t + \gamma r_{t+1} + ... + \gamma^{n-1-t} r_{n-1}$
    - Add these transition tuples with the corresponding cumulative reward ($q_t$) to a demonstration buffer. (The demonstration buffer should contain tuples of $(s_t, a_t, q_t, s_{t+1})$.)
    - Update the loss function for training the DDPG critic to be:<br>
    $\sum_{(s,a,r,s')}(Q_{rl} - Y)^2 + \lambda*\sum_{(s,a,q,s')}(Q_{rl} - \hat{Q})^2$<br>
    The first term is the bellman error, while the second term is a MSE loss w.r.t. demonstration data. $\hat{Q}$ is the expected cumulative reward calculated based on the demonstration trajectory.
    - With the above change to DDPG, we can explore the following ideas:
        1. Actively select the value of $\lambda$ based on some critics so that we can choose to learn more or less from demonstration.
        2. Consider the problem where the demonstrations can be noisy for some states.

    - Do not learn ungrounded Q value from demonstration.
    - The loss function is a n-step return -> under the assumption that the policy is near optimal, so mixed n-step return should work.
    - You should also use L2 regularization to prevent from overfitting to the small dataset
    - **Question**: How to make sure that the Q function prefers the action from demonstration after pre-training?

2. The uncertainty problem, how to combine Q_{RL} and Q_{Demo}
    - The target Q value should be a weighted combination of Q_{RL} and Q_{Demo}: Alpha*Q_{RL} + (1-Alpha)*Q_{Demo}
    - Implemented: Need calibration
        - Alpha = 1 as time -> infinity
        - Alpha ~= 0 as time -> 0

3. More environments/experiments to show that our current method is beneficial (High priority).
    - Since the uncertainty problem have not been solved. Currently, we have to assume that we are given an almost "perfect" function (s,a) -> Q, and prove that we can use this function to improve the learning performance of DDPG. For now, the "perfect" (s,a) -> Q is obtained from a well-trained ddpg critic.
    - Besides the two OpenAI environments,
        - Extra environment 1: peg in hole problem (either through modifying the openai robotics envs or using the robosuite peg in hole environment)
        - Extra env 2: navigation problem, e.g. the  OpenAI CarRacing environment.

4. How to acquire the training set of the demonstration neural net? Or, how to acquire or ask for a demonstration?
    1. Random:
        1. Before training the RL agent, select some states "s", and ask the expert for the expected corresponding action "a" for these states as well as the expected "Q".
        2. Before training the RL agent, select some states and action pairs (s, a), and ask the expert for the expected "Q" for these state action pairs.
    2. Active: (Train RL and Demonstration NN alternatively)
        1. During the RL agent training, based the uncertainty of the output from the critic function in DDPG and the uncertainty from the demonstration neural net, periodically select some states and action pairs (s, a) and ask the expert for the expected "Q" for these state action pairs.

## Topics for the meeting
### Experiments on Reach2DFO environment.
1. Assume the demonstration neural network is perfect, then how well can we do?
    - Use a well trained critic and change other experiment variables.
    - The well trained critic needs to have some uncertainty.
2. Test with various of environments. No significant improve. See exp results.

Extras...

1. We can have the weight that is purely based on the uncertainty of the demonstration neural net. -> basically what I am doing now
2. Forget about the greater sign?
3. We can even get rid of the demonstration nn?
    - Just require demonstration of the desired Q value of transition pairs drawn from the replay buffer (based on some condition such as uncertainty from the bootstrapped ensemble of critic.)

## Meeting Minutes
- Chernoff bnd width
- Chebyshev inequality
- Markov decision process



## Reminder