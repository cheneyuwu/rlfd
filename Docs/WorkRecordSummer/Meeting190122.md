# Undergrad Research Project
Meeting Summary<br>
January 22, 2019

## Thoughts on Existing Methods

- Replay Buffer, sample with priority
  - Idea: 
    - Similar idea as VIME
    - VIME calculates information gain for their model approximator (a BNN) through transition tuples $(s_t,a_t,s_{t+1})$
    - Similarly, we can calculate the information gain for the Q function in DDPG through transition tuples $(s_t,a_t,s_{t+1},r)$.
    - Then, when sampling from the replay buffer, prioritize tuples with high information gain.
  - Current Problem and Solutions
    - Consider the target equation that we want to minimize (Bellman equation): $\underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
          \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a') \right) \Bigg)^2
          \right]$  
      We cannot assume that bots $Q_\phi(s,a)$ and $max_{a'}Q_\phi(s',a')$ are normally distributed. This requires some derivation.  
      Notice that $V(s')=max_{a'}Q_\phi(s',a')$ -> can we do a gaussian process here by assuming for each fixed $a'$, $Q_\phi(s',a')$ is normally distributed.  
      Only if we can prove this, the following formula can hold:  
      &nbsp; $Q^{\frac{1}{N}\sum w_i}=r+\gamma E_{s'}[V_{t+1}(s')^{\frac{1}{N}\sum w_i}]$  
      Otherwise, we should consider sampling from the distribution and train on each sampled version of the Q function:  
      &nbsp; $Q^{w_i}=r+\gamma E_{s'}[V_{t+1}(s')^{w_i}]$  
      -  TODO: try to derive the formula and try with some really simple environments.
      -  Useful resources:
         -  Distributional Bellman and the C51 Algorithm
         -  A Distributional Perspective on RL (paper)
         -  Pilco: A model-based and data-efficient approach to policy search
         -  Distributed distributional deterministic policy gradients by DeepMind
         -  Distributional advantage actor-critic
         -  Distributional RL projected in Gaussian Space -> Gaussian Processes in RL (NIPS proceeding) and Nonparametric return distribution approximation for RL
         -  Cooperative inverse reinforcement learning 

- Simple improvement on current TD3 algorithm
  - Idea: 
    - Add some demonstration data and train on those data
    - The demonstration data should contain $(s_t,a_t, s_{t+1},Q_D)$
    - When training on the demonstration data we should generate $Q_{E1}$ and $Q_{E2}$ for each tuple, then choose $max(Q_D, min(Q_{E1},Q_{E2}))$.
  - Current Problem and Solutions
    - Even if the time horizon can be infinite, the demonstration data is finite. How do we calculate the Q then?
      - Make the time horizon to be finite. Stop as soon as it hits the goal. -> This may require some change to the replay buffer and the generate rollout worker.
  
- Variance of demonstration data and estimated data
  - Idea
    - Suppose that we replace Q function in DDPG with a BNN so that we can get the variance of the estimated cost-to-go, $Q_E$. Also suppose that we can get the variance of cost-to-go for our demonstration data, $Q_D$. We can use it to evaluate the demonstration data.
      - When $Q_D > Q_E$, and $Q_D$ has low variance.
        - We are almost sure that $Q_E$ is an under estimate.
      - When $Q_E > Q_D$
        - This is hard to say. We do not know if $Q_E$ is over estimating or $Q_D$ is sub-optimal
  - Problem and Solutions
    - Notice that if we every use samples from the BNN, we should re-consider this. Calculate $P(Q_1 > Q_2) >$ some certainty.

## New Thoughts
- Sequence of tasks
  - Suppose that a task can be splitted to sub tasks ABCD, then the variance should increase when going from A to D. Can we do something with the variance information in this case?

## Extra Notes
  - Set up slack

