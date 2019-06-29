Add all tunable params here!

# Demo Shaping
- whether or not to use regularizer and its weight
- scale of the potential (same as scale of the reward)
- use stochastic gd to train?
- whether or not to add noise to state and take average

# TD3
- Double critic network
- Noise to target actor
- Update actor at the half updating speed as critic


# Change in the latest commit:
- cmd changes for running on cluster
- use train_blockreach.py as example
- not using random noise in demo
- add RLfD through BC
- add noise to state when calculating potential, no regularizer
- shaping rewrd for performance evaluation
- use envname + Dense convention
- a bug in return from log_prob, shape should be (?, 1) not (?)