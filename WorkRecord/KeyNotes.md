# Undergrad Research Project
Work Record<br>
January 6, 2018

## Highlight From Last Meeting
### Project Idea/Definition/Goal
- Non-determinism under reinforcement learning
  - Policy 
    - can be avoid using DDPG
  - Reward
    - deterministic in robot control environments
    - However, it can vary between tasks in multi-goal RL
  - State transitions
    - system dynamics can be non-deterministic - VIME tried to minimize this
- [VIME](../Resources/Papers/VIME.md)
  - VIME cannot be used for multi-goal RL. For multigoal RL, we need to derive a similar formula for the $Q$ function.
  - Note: $Q(a_t, s_t) = r(s_t, a_t) + \gamma E_{s, p(\centerdot|s_t, a_t, \theta)}$
- [DDPG](../Resources/Papers/DDPG.md)
- Thoughts on applying VIME to DDPG
  - In VIME/Model-Based Policy Gradient, we learn the dynamic model $P(s_{t+1}|s_t ,a_t ,\theta)$ and choose a policy that is most informative for training this model. The dynamic moel is learnt by minimizing variational lower bound:
    - $L[q(\theta;\phi)||p(\theta)]=E_{\theta\sim q(\centerdot;\phi)}[log\ p(D|\theta)]-D_{KL}[q(\theta;\phi)||p(\theta)]$
  - In DDPG, instead of learning the model, we learn the Q function through Bellman Equation. This is done by minimizing the mean square Bellman error on tuples of 
    - $\underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
          \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a') \right) \Bigg)^2
          \right]$
  - We can apply the idea of learning the dynamic model in VIME to learning the Q function in DDPG, in which case the $D$ in the first term of variational lower bound formula will be the Bellman target: $\left(r + \gamma\max_{a'} Q_{\phi}(s',a') \right)$ .
  - For the KL divergence, we can calculate in the same way as which discussed in VIME.
  - The next problem is how we can use this KL divergence to improve exploration. In VIME/Policy Gradient, we calculate the KL divergence for every transition $(s_t,a_t,s_{t+1})$ and add it to the immediate reward $r_t$. However, in DDPG, the policy is trained to maximize output of the Q function directly, so we need to take use he KL divergence in a different way.
    1. As a separate step, train the policy to maximize $log\ P(\pi(s,g))*D_{KL}[q(\theta;\phi)||p(\theta)]$. Since the KL divergence is always non-negative, we should use relative KL divergence.
    2. In multi-goal, we can use KL divergence to determine which goals are relatively unexplored, and perform more extensive training on those goals.
    3. Maybe we can use KL divergence and Reward to determine whether the agent has learnt a task or the task is too hard to learn.  
       High reward + Low $D_{KL}$ would indicate that the task has been learnt.
       Low reward + Low $D_{KL}$ would indicate that the task is too hard to learn.

### Simulators and Environments
- Robotics environments of OpenAi
  - use this environment for initial testing
  - can compare the result directly with results from OpenAI
- Robosuite environment
  - Later, build the peg in hole environment by extending the robosuite environment