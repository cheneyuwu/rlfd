# Undergrad Research Project
Meeting Summary<br>
January 08, 2019

## Thoughts on Existing Methods
- Bootstrapped DQN
  - Compare to TD3, we may want to use multiple actors. Each actors corresponding to one Q function. However, I am not sure how much we can get from this change.
- VIME
  - As a separate step, train the policy to maximize $log\ P(\pi(s,g))*D_{KL}[q(\theta;\phi)||p(\theta)]$. Since the KL divergence is always non-negative, we should use relative KL divergence.
  - In multi-goal, we can use KL divergence to determine which goals are relatively unexplored, and perform more extensive training on those goals.
  - Maybe we can use KL divergence and Reward to determine whether the agent has learnt a task or the task is too hard to learn.  
       High reward + Low $D_{KL}$ would indicate that the task has been learnt.
       Low reward + Low $D_{KL}$ would indicate that the task is too hard to learn.

## Project Idea/Definition/Goal
- Replay Buffer, sample with priority
  - Similar idea as VIME
  - VIME calculates information gain for their model approximator (a BNN) through transition tuples $(s_t,a_t,s_{t+1})$
  - Similarly, we can calculate the information gain for the Q function in DDPG through transition tuples $(s_t,a_t,s_{t+1},r)$.
  - Then, when sampling from the replay buffer, prioritize tuples with high information gain.
- Simple improvement on current TD3 algorithm
  - Add some demonstration data and train on those data
  - The demonstration data should contain $(s_t,a_t, s_{t+1},Q_D)$
  - When training on the demonstration data we should generate $Q_{E1}$ and $Q_{E2}$ for each tuple, then choose $max(Q_D, min(Q_{E1},Q_{E2}))$.
- Variance of demonstration data and estimated data
  - Suppose that we replace Q function in DDPG with a BNN so that we can get the variance of the estimated cost-to-go, $Q_E$. Also suppose that we can get the variance of cost-to-go for our demonstration data, $Q_D$. We can use it to evaluate the demonstration data.
    - When $Q_D > Q_E$, and $Q_D$ has low variance.
      - We are almost sure that $Q_E$ is an under estimate.
    - When $Q_E > Q_D$
      - This is hard to say. We do not know if $Q_E$ is over estimating or $Q_D$ is sub-optimal
- Sequence of tasks
  - Suppose that a task can be splitted to sub tasks ABCD, then the variance should increase when going from A to D. Can we do something with the variance information in this case?

## Extra Notes
  - Set up slack

