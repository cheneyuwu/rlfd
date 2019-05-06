# Undergrad Research Project
Meeting Summary<br>
February 26, 2019

## TODO before meeting
- Find a way to get $Q_D$ for Mujoco environments, this may help for generating useful demonstration data.
    - Its time horizon can either be finite or infinite.
    - You can do this by train a RL agent first and get demo data from that pre-trained RL agent.
- Supervised learning for demonstration. Use Edward for BNN or Ensemble.
- Code for comparing distributions.

## Notes

### Idea 1: Add demonstration to RL + uncertainty
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
        - Throught supervise learning. The input data should be tuples of $(s, a, s', r, Q_D)$ and the output should be $\overset{\sim}{Q_D}$ for any transition tuple. If you use an ensemble of nns or a BNN (variantional inference), you should be able to obtain uncertainty for transitions in demonstration.
    - Compare distributions?
        - After we get uncertainties in RL and Demo, we should be able to calculate $P(Q_D > Q_{RL})$ and use it as $W$, the actual $Q$ should be something like: $Q = (1-W)E[Q_{RL}] + W E[Q_D]$
        - You can assume Gaussian distributions!
- Questions:    
    - In OpenAI's ddpg implementation, there is a Normalizer that normalizes observation and goals. However, with demonstration data, you need to disable this
    - because demonstration data may use a normalizer with different mean and variance.
        - How to solve this problem? 
            - maybe train the demo network and ddpg network at the same time? 
        - Should also make sure that any trick that could possibly change input or output should be reconsidered.
            - For example, clip observations, max_u?