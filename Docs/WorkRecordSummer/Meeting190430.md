# Undergrad Research Project
Meeting Summary<br>
April 30th, 2019

## Goal/Thoughts/Idea from last week
1. The uncertainty problem, how to combine Q_{RL} and Q_{Demo}
    - The target Q value should be a weighted combination of Q_{RL} and Q_{Demo}: Alpha*Q_{RL} + (1-Alpha)*Q_{Demo}
    - Implemented: Need calibration
        - Alpha = 1 as time -> infinity
        - Alpha ~= 0 as time -> 0
    - How should we calibrate Alpha based on the uncertainty of Q_{RL} and Q_{Demo}? (High Priority)

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

## Topics for the meeting
### Experiments on Reach2DFO environment.

1. Implemented No.1 from above.
2. Worked on the most simple environment. However, on more complex environments, the effect of wrong results are still large. We need to add a hard threshold?
    1. However, it is very likely that the uncertainty of demonstration neural net can go large and then RL learns nothing from the demonstration nn.

Extras...

1. We can have the weight that is purely based on the uncertainty of the demonstration neural net. -> basically what I am doing now
2. Forget about the greater sign?
3. We can even get rid of the demonstration nn?
    - Just require demonstration of the desired Q value of transition pairs drawn from the replay buffer (based on some condition such as uncertainty from the bootstrapped ensemble of critic.)

## Meeting Minutes
- Chernoff bnd width
- Chebyshev inequality
- Markov decision process



## Reminder