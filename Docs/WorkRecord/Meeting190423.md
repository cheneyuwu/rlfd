# Undergrad Research Project
# Undergrad Research Project
Meeting Summary<br>
April 17th, 2019

## Goal/Thoughts/Idea from last week
1. Must verify with multiple environments that, if the uncertainty works, everything afterwards should work.
    - Try using the environments provided in the OpenAI envs. Should also look for other environments.
2. Some other methods that's worth a try to train the demonstration network.
    - Only add the Q estimation of the initial (s,a) pair. Do not add the entire episode.
    - Train each demonstration head with different data set. Sample with replacement.
3. Should also consider what to consider when weighting the output from the critic and from the demonstration network
    - Model uncertainty of the demonstration network
    - Time -> should gradually abandon the output from demonstration
    - Need a way to require some knowledge about whether a (s,a) pair has been seen in the training set of the demonstration neural net.

## Topics for the meeting
### Experiments on Reach2DFO environment.
1. Only add the Q estimation of the initial (s,a) pair. Do not add the entire episode. -> helped with the demonstration nn training
2. Train each demonstration head with different data set. Sample with replacement. -> No significant effect
3. Looked into the uncertainty provided by the bootstrapped ensemble trained with expert data -> Overall uncertainty looks good. May need to decide a safe weight metric


## Meeting Minutes
- Chernoff bnd width
- Chebyshev inequality
- Markov decision process
-

1. The uncertainty problem, how to combine Q_{RL} and Q_{Demo}
    - The target Q value should be a weighted combination of Q_{RL} and Q_{Demo}: Alpha*Q_{RL} + (1-Alpha)*Q_{Demo}
    - We have agreed that
        - Alpha = 1 as time -> infinity
        - Alpha ~= 0 as time -> 0
    - How should we calibrate Alpha based on the uncertainty of Q_{RL} and Q_{Demo}? (High Priority)
        - We talked a lot about this last time and it was getting complex at the end. I am not sure if I understood you completely. I am currently looking for more resources about this and trying to get more thoughts. I will write down more notes on this and discuss with you through another email or through slack later.

2. More environments/experiments to show that our current method is beneficial (High priority).
    - Since the uncertainty problem have not been solved. Currently, we have to assume that we are given an almost "perfect" function (s,a) -> Q, and prove that we can use this function to improve the learning performance of DDPG. For now, the "perfect" (s,a) -> Q is obtained from a well-trained ddpg critic.
    - Besides the two OpenAI environments,
        - Extra environment 1: peg in hole problem (either through modifying the openai robotics envs or using the robosuite peg in hole environment)
        - Extra env 2: navigation problem, e.g. the  OpenAI CarRacing environment.

3. How to acquire the training set of the demonstration neural net? Or, how to acquire or ask for a demonstration?
    - I used to worry a lot about this. But it should have a lower priority than the uncertainty issue.
    - Currently we have discussed two strategies:
        1. Random:
           1. Before training the RL agent, select some states "s", and ask the expert for the expected corresponding action "a" for these states as well as the expected "Q".
           2. Before training the RL agent, select some states and action pairs (s, a), and ask the expert for the expected "Q" for these state action pairs.
        2. Active: (Train RL and Demonstration NN alternatively)
           1. During the RL agent training, based the uncertainty of the output from the critic function in DDPG and the uncertainty from the demonstration neural net, periodically select some states and action pairs (s, a) and ask the expert for the expected "Q" for these state action pairs.

Forget about the greater sign.
We can have the weight that is purely based on the uncertainty of the demonstration neural net.
We can even get rid of the demonstration nn. Just require demonstration of the desired Q value of transition pairs drawn from the replay buffer (based on some condition such as uncertainty from the bootstrapped ensemble of critic.)

## Reminder