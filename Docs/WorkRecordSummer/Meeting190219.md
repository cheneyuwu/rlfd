# Undergrad Research Project
Meeting Summary<br>
February 19, 2019

## TODO before meeting
- Implement ensemble of Q function idea and test it on simple environments
    - Looks like that ensemble of Q functions does help exploration, even if we do not use it to help for deciding demonstration
- Get some demonstration data for the Reach1D environment with Q value
    - Note that you only need Q values for demonstration data. for Others, just use a small value? because demonstration data does not need to be optimal
    - in general rollout, you need to store reward and then use the reward to calculate Q value!
- Implement D4PG

## Notes
- How can we have uncertainty in demonstration data?
    - Throught supervise learning. The input data should be tuples of $(s, a, s', r, Q_D)$ and the output should be $\overset{\sim}{Q_D}$ for any transition tuple. If you use an ensemble of nns or a BNN (variantional inference), you should be able to obtain uncertainty for transitions in demonstration.
- Compare distributions?
    - After we get uncertainties in RL and Demo, we should be able to calculate $P(Q_D > Q_{RL})$ and use it as $W$, the actual $Q$ should be something like: $Q = (1-W)E[Q_{RL}] + W E[Q_D]$
    - You can assume Gaussian distributions!
  - Prioritised experience replay? -- hold this off
- Conflict b/t td3 and bootstrapped DDPG.
    - There is no conflict between ensemble of AC and td3 algorithm. However, for now just focus on DDPG.