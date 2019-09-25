# Undergrad Research Project
Meeting Summary<br>
December 21, 2018

## Project Idea/Definition/Goal
  - Non-determinism under reinforcement learning
    - Policy 
      - can be avoid using DDPG
    - Reward
      - usually deterministic in robot control environments
      - will vary between tasks in the context of multi-goal RL
    - State transitions
      - system dynamics can be non-deterministic
      - This is what VIME tried to minimize
  - Idea of VIME, a model based method
    - $P_\alpha(\tau)=P(s_0)\Pi P(s_{t+1}|s_t ,a_t ,\theta)\Pi \pi(a_t|s_t,\alpha)$
    - Using DDPG, policy $\pi$ is deterministic so that the only non-deterministism happens in state trasition. VIME tries to choose a policy such that the transition probability (s_t, a_t, s_{t+1}) are most informative for training the dynamic model.
    - However, VIME cannot be used for multi-goal RL. For multigoal RL, we need to derive a similar formula for the reward function $Q$.
    - $Q(a_t, s_t) = r(s_t, a_t) + \gamma E_{s, p(\centerdot|s_t, a_t, \theta)}$
  - Multigoal RL problem setup
    - For the policy:
      - Input: state/observation and the goal for this entire episode
      - Ouput: next action
    - For the reward function
      - Input: state, action and the goal for this entire episode
      - Ouput: expected cumulated reward

## Simulators and Environments
  - Robotics environments of OpenAi
    - use this environment for initial testing
    - can compare the result directly with results from OpenAI
  - Robosuite environment
    - Build the peg in hole environment by extending the robosuite environment (if necessary)
  - PhysX from Nvidia
    - Since it is c++ based. Let's leave this for now.

## Extra Notes
  - Always use dense reward rather than sparse reward.

