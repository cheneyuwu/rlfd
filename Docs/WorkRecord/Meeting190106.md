# Undergrad Research Project
Work Record<br>
January 6, 2019

## Highlight From Last Meeting
### Project Idea/Definition/Goal
- Non-determinism under reinforcement learning
  - Policy 
    - can be avoid using DDPG
  - Reward
    - usually deterministic in robot control environments
    - will vary between tasks in multi-goal RL
  - State transitions
    - system dynamics can be non-deterministic - VIME tried to minimize this
- Idea of VIME
  - $P_\alpha(\tau)=P(s_0)\Pi P(s_{t+1}|s_t ,a_t ,\theta)\Pi \pi(a_t|s_t,\alpha)$
  - Using DDPG, policy $\pi$ is deterministic so that the only non-deterministism happens in state trasition. VIME tries to choose a policy such that the transition probability (s_t, a_t, s_{t+1}) are most informative for training the dynamic model.
  - However, VIME cannot be used for multi-goal RL. For multigoal RL, we need to derive a similar formula for the $Q$ function.
  - $Q(a_t, s_t) = r(s_t, a_t) + \gamma E_{s, p(\centerdot|s_t, a_t, \theta)}$
- Multigoal RL problem setup
  - For the policy:
    - Input: state/observation and the goal for this entire episode
    - Ouput: next action
  - For the reward function
    - Input: state, action and the goal for this entire episode
    - Ouput: expected cumulated reward
### Simulators and Environments
- Robotics environments of OpenAi
  - use this environment for initial testing
  - can compare the result directly with results from OpenAI
- Robosuite environment
  - Later, build the peg in hole environment by extending the robosuite environment

## Summary of Last Week
- For Bootstrapped DQN:
  - Last time you mentioned that this method looks similar to the one used in this paper: https://arxiv.org/pdf/1802.09477.pdf that proposed TD3 (Twin delayed DDPG). Later I looked into their algorithm, and found that the algorithm they proposed was actually different from what we would come up for our idea and goal. The main purpose of their modification to DDPG is to avoid overestimation bias in Q function; However, our goal is to maintain a distribution of Q values through ensemble of Q functions (like Bootstrapped DQN) or BNN (VIME) for effective exploration. In their method, they use 1 actor and 2 critics and the critics also share 1 replay buffer. However, if we use ensemble of Q functions in DDPG, I think we should also use multiple actors (same as number of critics) and train each critic on a different data set or a subset of the replay buffer. Let's discuss the difference of the ideas in these two papers in more detail later.
- For VIME:
  - I spent some time looking into they derivation and implementations. As discussed last time, if we want to use VIME's idea to improve DDPG, we should calculate the information gain of critic from each transition tuple $(s_t, goal, a_t, s_{t+1})$ and make use of this info gain to enable active exploration or as a estimation of the difficulty of the task. What I found when applying VIME to DDPG was that, the formula for calculating info gain should not change much; However, the main problem is how we can make use of this information gain. I want to discuss with you about if the formula I thought about is correct and how we can use it in DDPG.

## Details
### Summary of the Existing DDPG Implementation, Idea of VIME and Their Connections
- [VIME](###VIME)
  - VIME cannot be used for multi-goal RL. For multigoal RL, we need to derive a similar formula for the $Q$ function.
  - Note: $Q(a_t, s_t) = r(s_t, a_t) + \gamma E_{s, p(\centerdot|s_t, a_t, \theta)}$
- [DDPG](###DDPG)
- Thoughts on applying VIME to DDPG
  - In VIME/Model-Based Policy Gradient, we learn the dynamic model $P(s_{t+1}|s_t ,a_t ,\theta)$ and choose a policy that is most informative for training this model. The dynamic model is learnt by minimizing variational lower bound:
    - $L[q(\theta;\phi)||p(\theta)]=E_{\theta\sim q(\centerdot;\phi)}[log\ p(D|\theta)]-D_{KL}[q(\theta;\phi)||p(\theta)]$
  - In DDPG, instead of learning the model, we learn the Q function through Bellman Equation. This is done by minimizing the mean square Bellman error on tuples of 
    - $\underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
          \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a') \right) \Bigg)^2
          \right]$
  - We can apply the idea of learning the dynamic model in VIME to learning the Q function in DDPG, in which case the $D$ in the first term of variational lower bound formula will be the Bellman target: $\left(r + \gamma\max_{a'} Q_{\phi}(s',a') \right)$ .
  - For the KL divergence, we can calculate in the same way as which discussed in VIME.
  - The next problem is how we can use this KL divergence to improve exploration. In VIME/Policy Gradient, we calculate the KL divergence for every transition $(s_t,a_t,s_{t+1})$ and add it to the immediate reward $r_t$. However, in DDPG, the policy is trained to maximize output of the Q function directly, so we need to take use he KL divergence in a different way.
    1. As a separate step, train the policy to maximize $log\ P(\pi(s,g))*D_{KL}[q(\theta;\phi)||p(\theta)]$. Since the KL divergence is always non-negative, we should use relative KL divergence.
    2. In multi-goal RL, we can use KL divergence to determine which goals are relatively unexplored, and perform more extensive training on those goals.
    3. Maybe we can use KL divergence and Reward to determine whether the agent has learnt a task or the task is too hard to learn.  
       High reward + Low $D_{KL}$ would indicate that the task has been learnt.
       Low reward + Low $D_{KL}$ would indicate that the task is too hard to learn.

### VIME
#### Key Notes and Equations
- $P_\alpha(\tau)=P(s_0)\Pi P(s_{t+1}|s_t ,a_t ,\theta)\Pi \pi(a_t|s_t,\alpha)$
- Using DDPG, policy $\pi$ is deterministic so that the only non-deterministism happens in state trasition. VIME tries to choose a policy such that the transition probability ($s_t, a_t, s_{t+1}$) are most informative for training the dynamic model.
- State transition model: $\int p(s_{t+1}|s_t, a_t; \theta)p(\theta)$
- Maximizing the sum of reductions in entropy (uncertainty)
  - $\sum_t(H(\Theta|\xi_t,a_t)-H(\Theta|S_{t+1},a_t))=\sum_tI(S_{t+1};\Theta|\xi_t,a_t)$
  - $\sum_tI(S_{t+1};\Theta|\xi_t,a_t)=E_{s_{t+1}~P(\centerdot | \xi_t, a_t)}[D_{KL}[p(\theta|\xi_t,a_t,s_{t+1})||p(\theta|\xi_t)]]$
- Bayes' rule
  - $p(\theta|\xi_t,a_t,s_{t+1})=\dfrac{p(\theta|\xi_t)p(s_{t+1}|\xi_t,a_t;\theta)}{p(s_{t+1}|\xi_t,a_t)}$
  - This is difficult to compute in high dimensional space.
- Variational inference
  - Approximate $p(\theta|D)$ through an alternative distribution $q(\theta;\phi)$, parameterized by $\phi$, and minimize $D_{KL}[q(\theta;\phi)||p(\theta|D)]$.
  - Maximize the variational lower bound $L[q(\theta;\phi)||p(\theta)]$
    - $L[q(\theta;\phi)||p(\theta)]=E_{\theta\sim q(\centerdot;\phi)}[logp(D|\theta)]-D_{KL}[q(\theta;\phi)||p(\theta)]$
    - The first term is log likelihood
    - The second term is distance to **prior**
- Update rules under the config of Gaussian distribution
  - Model - BNN 
    - $q(\theta; \phi)=\Pi_{i=1}^{|\Theta|}\N(\theta_i|\mu_i;\sigma_i^2)$ with $\phi=\{\mu,\sigma\}$
    - Also use $\sigma=log(1+e^\rho)$
  - Maximize variational lower bound $L[q(\theta;\phi)||p(\theta)]$
    - $\phi' =\underset{\phi}{arg\ min}[D_{KL}[q(\theta;\phi||q(\theta;\phi_{t-1}))]-E_{\theta \sim q(\centerdot ; \phi)}[log\ p(s_t|\xi_t,a_t;\theta)]]$
    - Note 1: $E_{\theta\sim q(\centerdot;\phi)}[logp(D|\theta)] \approx \dfrac{1}{N}\sum ^N_{i=1}logp(D|\theta_i)$ with $N$ samples drawn according to $\theta \sim q(\centerdot ; \phi)$
  - Compute $D_{KL}[q(\theta;\phi')||q(\theta;\phi)]$ by approximating $\nabla^TH^{-1}\nabla$ of the above for each $(s_t,a_t,s_{t+1})$ tuple generated during rollout.
- Modified reward for policy:
    - $r\prime(s_t,a_t,s_{t+1})=r(s_t,a_t)+\eta D_{KL}[q(\theta;\phi_{t+1}||q(\theta;\phi_t))]$, where $\phi_{t+1}$ is the prior belief, $\phi_t$ is the posterior reward.

### DDPG
#### Key Notes and Equations
  - Q learning side
    - Input: state/observation $s$, goal of this episode $g$ and action $a$
    - Bellman equation
      - $Q^*(s,a)=\underset{s' \sim P}{E}[r(s,a)+\gamma \underset{a'}{max}Q^*(s',a')]$
      - $s'$ and $a'$ are the next state and action, respectively.
    - Mean-squared Bellman error (MSBE)
      - $L(\phi, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
          \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a') \right) \Bigg)^2
          \right]$
      - $d$ to indicate whether $s'$ is a terminal state (not used in my implementation)
    - Trick 1: replay buffer that stores tuple of $(s,a,r,s',d)$
    - Trick 2: target networks
      - Target: $r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a')$
      - Polyas averaging: $\phi_{\text{targ}} \leftarrow \rho \phi_{\text{targ}} + (1 - \rho) \phi$
    - Loss function:
      - $L(\phi, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
        \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) Q_{\phi_{\text{targ}}}(s', \mu_{\theta_{\text{targ}}}(s')) \right) \Bigg)^2
        \right]$
  - Policy learning side, gradient ascent to maximize Q value.
    - $\max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi}(s, \mu_{\theta}(s)) \right]$
  - Exploration vs. Exploitation
    - $\epsilon$-greedy
    - uncorrelated, mean-zero Gaussian noise