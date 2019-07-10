# RL + IL Research Project

## Problem: Consider non-determinism under reinforcement learning
- Non-Determinism in RL Policy
    - can be avoid using DDPG
- Non-Determinism in Reward
    - The reward is usually deterministic in robot control environments
    - However, it can vary between tasks in multi-goal RL -> but the goal can also be considered as state.
- State transitions
    - System dynamics can be non-deterministic - The VIME paper tried to minimize this non-determinism

## Goal: Active Selection in DDPG
- In DDPG, we learn the Q function through Bellman Equation, which is done by minimizing the mean square Bellman error on tuples of<br>
  $\underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[\Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a') \right) \Bigg)^2\right]$

## Idea 1: Combine IL and RL by training DDPG critic using demonstration data.
- Generate some demonstration trajectories $(s_1, a_1, r_1, s_2, a_2, r_2, ...)$.
- Since you have the entire trajectory, you can calculate the expected cumulative reward for each transition tuple $(s_t, a_t)$ along the trajectory.
- Add these transition tuples with the corresponding cumulative reward ($q_t$) to a demonstration buffer. (The demonstration buffer should contain tuples of $(s_t, a_t, q_t, s_{t+1})$.)
- Update the loss function for training the DDPG critic to be:<br>
  $\sum_{(s,a,r,s')}(Q_{rl} - Y)^2 + \lambda*\sum_{(s,a,q,s')}(Q_{rl} - \hat{Q})^2$<br>
  The first term is the bellman error, while the second term is a MSE loss w.r.t. demonstration data. $\hat{Q}$ is the expected cumulative reward calculated based on the demonstration trajectory.
- With the above change to DDPG, we can explore the following ideas:
    1. Actively select the value of $\lambda$ based on some critics so that we can choose to learn more or less from demonstration.
    2. Consider the problem where the demonstrations can be noisy for some states.

## Idea 2: Replay buffer, sample with priority (lots of papers on this idea)
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
- Implementation:
    1. BNN solution - not going towards this because the formula is hard to derive.
    2. Ensemble of Q functions
        - Implement bootstrapped DDPG and try to create a replay buffer that prioritizes tuples with high uncertainty in the Q value output from each critic.

## Idea 3: Train the DDPG critic using demonstraion data weighting on its uncertainty (Did not work well so lower its priority.)
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

- Current Status
    - Now we focus on Ensemble for both RL and Demonstration. We compare 1 pair of actor critic with the distribution of Q from demonstration.
    - Priority 1: Proof required: $P(E[Q]|D, s, a)$ is normally distributed?
        - Different type of uncertainties to consider.
            1. Uncertainties in the Q value for RL.
                - For each transition tuple, Q forms an unknown distribution. Distributional RL learns this distribution for each transition tuple, traditional DDPG learns the expected Q value.
            2. Uncertainties in the Q value from Demonstration
                - Similarly, when learning the demonstration Q directly, we also have uncertainties in demonstration Q whose distribution is unknown. For example, in different trajectories, we may have different Q value even for the same (s,a,s').
                - In this case, can we say that we are learning the expected Q value? Yes, because we are using the quadratic loss function. And it will converge to the mean value of the Q.
            3. How to prove that the E(Q) from different network instances form normal distribution? It does not really follows the central limit theorem because the central limit theorem requires drawing from the same distribution. However, according to the paper "Deep Neural Networks as Gaussian Processes", we probably can assume that it forms a normal distribution.
    - Priority 2: Find environments that could benefit from the current setting.
        - For positive reward: this method may break exploration, because it notices that everything looks better than the current estimation. So it will follow one path and never explore others. I think the essential problem here is that it breaks exploration so that the actor never learns the correct actions.
        - If the demo Q value is much smaller than the true Q value. This still does not converge. We can see that from the NReward result.
    - Priority 3
        - use $wQ_D + (1-w)Q_E$
    - Priority 4: Use BC to learn a policy that has the same input and output as the actor.
        - The demo actor is just used to directly learn from demonstration. It tries to learn, not memorize the behavior from the demonstration.
        - Thought 1: If we just consider this guided NN as one bootstrapped head. Then this is very similar to adding demonstration data to the replay buffer of a bootstrapped ddpg, although it may be useful for guided exploration.
        - Thought 2: If we completely separate the demo actor and rl actor. (Separate? never train the demo actor again? Just use it for comparison?) There are existing methods that use let the actor selectly clone the behavior of the demo actor based on a q filter.
        - Thought 3: we need to take the uncertainty of the actor into account. How?
        - Note: This agent can be a nn or decision tree or gaussian process. We need to take the uncertainty of it into account. Refer to the *Improving Reinforcement Learning with Confidence-Based Demonstrations* paper (i.e. the C-Hat algorithm).
        - Questions to consider:
            1. whether or not should we train the demonstration NN during rl
            2. whether or not should we have multiple nns for the demonstration. -> Since we need to consider the uncertainty of the demonstration, we definitely should.
               1. If so, what does the uncertainty mean in this case? Should we train them on the same data? or not? According to bootstrapped DQN, this probably won't matter that much. However, we should still consider this.
            3. How can we combine this with active learning? should we ask for demonstration when the uncertainty of rl actor is high or when the uncertainty of the demo actor is high? or both?
    - Extra note: there is a paper developed ARfD, which actively request for demonstration based on the uncertainty of the Q value from bootstrapped Q learning. This is the same idea as what we had a few months ago. Refer to LearningFromDemo.md and Exploration.md for more information.

## Idea 4 Sequence of tasks
- Suppose that a task can be splitted to sub tasks ABCD, then the variance should increase when going from A to D. Can we do something with the variance information in this case?

## Resources (To be moved to another directory)
- Pilco, Gaussian Processes in RL,
    - Model based methods, Model the systems using GP
- Non-parametric return distribution approximation for RL
- Distributional advantage actor-critic
    - Similar idea as C51

## Simulators and Environments
- Robotics environments of OpenAi
    - use this environment for initial testing
    - can compare the result directly with results from OpenAI
- Robosuite environment
    - Later, build the peg in hole environment by extending the robosuite environment