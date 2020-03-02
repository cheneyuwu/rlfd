# RL + IL Research Project

## High Level Problem: Consider non-determinism under reinforcement learning
- Non-Determinism in RL Policy
    - can be avoid using DDPG
- Non-Determinism in Reward
    - The reward is usually deterministic in robot control environments
    - However, it can vary between tasks in multi-goal RL -> but the goal can also be considered as state.
- State transitions
    - System dynamics can be non-deterministic - The VIME paper tried to minimize this non-determinism

## Idea 1: Learning from demonstrations through shaping
- Potential / Shaping:
    - Use a potential function, which is a normalizing flow/GAN
- Demonstrations
    - The demonstration data is considered to have no reward signal
    - Put demonstrations in the replay buffer
    - Initialize the policy with demonstrations (BC)
- RL side
    - DDPG / SAC
    - Try multistep return, priortized replay
- Environments to Test
    1. GYM Pick and Place with fixed goal
    2. GYM Peg in Hole with fixed goal
    3. MetaWorld Press Button

### Progress
- Paper
- Thesis Paper

### Contribution:
- In order to understand the paper’s contribution, it would have helped to know that it is an application of the insight presented in [11] in combination with applying “recent” generative models such as GANs and normalizing flows.

### Related work
Suboptimal demonstrations are included by these RL approaches:
- “Overcoming Exploration in Reinforcement Learning with Demonstrations” by Nair et al.
- “Reinforcement Learning from Imperfect Demonstrations” by Gao et al.
- “Interactive Reinforcement Learning with Dynamic Reuse of Prior Knowledge from Human/Agent's Demonstration” by Wang et al.
- “Truncated Horizon Policy Search: Combining Reinforcement Learning & Imitation Learning” by Sun et al.
- “RL with supervision by a stable controller” by Rosenstein et al.
- “Residual RL for Robot Control” by Johannink et al.
- “Residual Policy Learning” by Silver et al.
    - Although one might argue that residual RL methods are irrelevant in this context, I think they are a reasonable alternative to deal with sub-optimal demonstrations and to avoid forgetting schedules:
- “Goal-conditioned Imitation Learning” Ding et al.
    - A few Imitation Learning approaches might be a more state-of-the-art comparison instead of pure Behavioral Cloning, e.g.:

- Pastor, Peter, et al. "Skill learning and task outcome prediction for manipulation." ICRA, 2011.
- Kormushev, Petar, at al. "Robot motor skill coordination with EM-based reinforcement learning." IROS, 2010.
- Kober, Jens, and Jan R. Peters. "Policy search for motor primitives in robotics." Advances in neural information processing systems. 2009.

These are more efficient than NN methods
- Chatzilygeroudis, Konstantinos, et al. "Black-box data-efficient policy search for robotics." IROS, 2017.
- Deisenroth, Marc, and Carl E. Rasmussen. "PILCO: A model-based and data-efficient approach to policy search." ICML, 2011.

Policy search methods
- Chatzilygeroudis et al. "A survey on policy search algorithms for learning robot controllers in a handful of trials"
- Deisenroth et al. "A Survey on Policy Search for Robotics"
    - Deep Learning methods are used solve problem that has been tackled with Policy Search Reinforcement Learning in a more data efficient fashion. Therefore, the convenient applicability of the approach to real robot systems is not fully evident. A deeper discussion and ideally a comparison is required on the advantage of deep-learning-based methods.

### Experiments
- What is the action space of the peg-insertion and pick-and-place task?
(From Fig. 3 it looks like a 2D action space?  I assume not, because in IV.B it is written that only(?) the sensitivity experiment is “limiting the state space to a 2D plane”.)
- Is the simulation purely kinematic or are forces considered?
- How is the picking action simulated? Does the end-effector only need to be in the vicinity of the object or do the fingers need to be actuated?
- How are the demonstrations generated?
- The argument against BC+\lambdaTD3 is its sensitivity to the hyperparameter \lambda (shown in Fig. 4). The proposed GAN shaping approach also contains hyperparameters, but no evaluation is shown. How sensitive is the proposed approach w.r.t. its hyperparameters?
- What is the tolerance in the peg-insertion task?
- Compare the performance of the proposed approach to the various methods presented in the past.

- Simplifications on real robots -> not easy
- What demonstrations are referred as sub-optimal?
- Replicate the real world experiment?
- Need more complicated environment for showing sub-optimality -> the 2D environment is not enough
- How many demonstrations are required in the algorithm?
- Shortest path metric is not the best metric for peg insertion

- Find some environments solvable by the baseline methods

- More complicated experiments

- Results in Section IV needs an introductory paragraph to describe the experimental concept and setup (data acquisition, data length both in simulation and real experiments, define the term “episodes” etc).
    - By now, figure plots of Section IV are not quite comprehensive, because of missing this description.

### Video
- It would be helpful to show the actual demonstrations.

### Typos & Expressions
- Overall, the paper is written clearly, a few comments:
    There is a confusing change of meaning of the lambda parameter. In Eq.
    2 lambda is a multiplier that relates to the demonstrations (higher
    lambda -> closer to demonstrations). In the experimental section lambda
    is a multiplier of TD3 (higher lambda -> closer to RL objective).
- What is the meaning of green/red in Fig. 3 “Demonstrations”? (it helps to explain it in the caption)
- “It can handle multi-modal demonstrations gracefully” (p. 1): This might be misunderstood since multi-modal demonstrations oftentimes refer to a multimodal state space. Maybe re-use “multi-modal action distributions”.
- The explanations in sections IIIB and IIIC seem close to the original papers. Are they needed? Specify application-specific insights that were learned here.
- P. 2: “real” &#8594; “realm”
- P. 3: “since we assume kinesthetic teaching and not high-dimensional image data” &#8594; These two things are orthogonal.
- P. 5: “environment dos not” &#8594; “does not”

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