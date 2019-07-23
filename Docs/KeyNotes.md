# RL + IL Research Project

## High Level Problem: Consider non-determinism under reinforcement learning
- Non-Determinism in RL Policy
    - can be avoid using DDPG
- Non-Determinism in Reward
    - The reward is usually deterministic in robot control environments
    - However, it can vary between tasks in multi-goal RL -> but the goal can also be considered as state.
- State transitions
    - System dynamics can be non-deterministic - The VIME paper tried to minimize this non-determinism

## High Level Goal: Active Selection in DDPG from 
- In DDPG, we learn the Q function through Bellman Equation, which is done by minimizing the mean square Bellman error on tuples of<br>
  $\underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[\Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a') \right) \Bigg)^2\right]$

## Idea 1: Learning from demonstrations through shaping
- use a potential function, which is a maf trained on masked auto regressive flow
- Environments to Test
1. Reach (Current)
    The initial state of the arm is fixed. The target position is also fixed.
2. Pick and Place
    The initial state of the arm, initial location of the object to be picked and the target position are fixed.
3. Peg in Hole
    The arm always holds the object.
    The initial state is fixed
    The hole location is fixed.
    Later we can change the shape of the hole to make the environment harder.
4. Stack(?)
    2 or 3 objects are put on the table, and the goal is to put one on another.

## Idea 2: Combine IL and RL by training DDPG critic using demonstration data.
- Generate some demonstration trajectories $(s_1, a_1, r_1, s_2, a_2, r_2, ...)$.
- Since you have the entire trajectory, you can calculate the expected cumulative reward for each transition tuple $(s_t, a_t)$ along the trajectory.
- Add these transition tuples with the corresponding cumulative reward ($q_t$) to a demonstration buffer. (The demonstration buffer should contain tuples of $(s_t, a_t, q_t, s_{t+1})$.)
- Update the loss function for training the DDPG critic to be:<br>
  $\sum_{(s,a,r,s')}(Q_{rl} - Y)^2 + \lambda*\sum_{(s,a,q,s')}(Q_{rl} - \hat{Q})^2$<br>
  The first term is the bellman error, while the second term is a MSE loss w.r.t. demonstration data. $\hat{Q}$ is the expected cumulative reward calculated based on the demonstration trajectory.
- With the above change to DDPG, we can explore the following ideas:
    1. Actively select the value of $\lambda$ based on some critics so that we can choose to learn more or less from demonstration.
    2. Consider the problem where the demonstrations can be noisy for some states.

## Idea 3: Replay buffer, sample with priority (lots of papers on this idea)
- Overview
    - The general idea is that, we can use {TD error, KL divergence or some technique} to determine which states are relatively unexplored, and perform more extensive training on those states.
        - For KL divergence: we can calculate the information gain for the Q function in DDPG through transition tuples $(s_t,a_t,s_{t+1},r)$.
    - Usually, we should create a prioritized replay buffer that prioritizes tuples with high information gain. If we use ensemble of Q functions, we should probably use the epistemic uncertainty from the output of critics.
- Resources:
    - Prioritized Experience Replay
        - Priority based on TD error + some tricks
    - D4PG
        - Distributional DDPG, similar idea as C51
        - Similar idea. It also prioritizes tuples with high TD errors.

## Idea 4: Train the DDPG critic using demonstraion data weighting on its uncertainty (Did not work well so lower its priority.)
- Get a distribution for Q estimation in RL through BNN or Bootstrapped ActorCritic, say $Q_E$.
- Get a distribution for Q estimation in demonstration through supervised learning using BNN or Ensemble of Q functions, say $Q_D$.
- We can probably use demonstration to help for training
    - When $Q_D > Q_E$, and $Q_D$ has low variance.
        - We are almost sure that $Q_E$ is an under estimate.
    - When $Q_E > Q_D$
        - This is hard to say. We do not know if $Q_E$ is over estimating or $Q_D$ is sub-optimal
- Thoughts:
    - Use Ensemble/BNN for RL and BNN/Ensemble for demonstration.
    - How can we have uncertainty in demonstration data?
        - Throught supervise learning. The input data should be tuples of $(s, a, s', r, Q_D)$ and the output should be $\overset{\sim}{Q}_D$ for any transition tuple. If you use an ensemble of nns or a BNN (variantional inference), you should be able to obtain uncertainty for transitions in demonstration.
    - Compare distributions?
        - After we get uncertainties in RL and Demo, we should be able to calculate $P(Q_D > Q_{RL})$ and use it as $W$, the actual $Q$ should be something like: $Q = (1-W)E[Q_{RL}] + W E[Q_D]$
        - You can assume Gaussian distributions! Can you?
    - BNN Solution
        - Notice that if we every use samples from the BNN, we should re-consider this. Calculate $P(Q_1 > Q_2) >$ with some certainty.
        - Cannot Assume Normal Distribution
            - Consider the target equation that we want to minimize (Bellman equation): $\underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
               \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a') \right) \Bigg)^2
               \right]$
            We cannot assume that bots $Q_\phi(s,a)$ and $max_{a'}Q_\phi(s',a')$ are normally distributed. This requires some derivation.
            Notice that $V(s')=max_{a'}Q_\phi(s',a')$ -> can we do a gaussian process here by assuming for each fixed $a'$, $Q_\phi(s',a')$ is normally distributed.
            Only if we can prove this, the following formula can hold:
            $Q^{\frac{1}{N}\sum w_i}=r+\gamma E_{s'}[V_{t+1}(s')^{\frac{1}{N}\sum w_i}]$
            Otherwise, we should consider sampling from the distribution and train on each sampled version of the Q function:
            $Q^{w_i}=r+\gamma E_{s'}[V_{t+1}(s')^{w_i}]$

    - Distributional RL
        - Distributional RL gives you aleatoric uncertainty.
    - Extra note: there is a paper developed ARfD, which actively request for demonstration based on the uncertainty of the Q value from bootstrapped Q learning. This is the same idea as what we had a few months ago. Refer to LearningFromDemo.md and Exploration.md for more information.

## Idea 5 Sequence of tasks
- Suppose that a task can be splitted to sub tasks ABCD, then the variance should increase when going from A to D. Can we do something with the variance information in this case?