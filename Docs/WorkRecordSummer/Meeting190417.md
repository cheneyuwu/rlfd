# Undergrad Research Project
# Undergrad Research Project
Meeting Summary<br>
April 17th, 2019

## Goal/Thoughts/Idea from last week
    1. We now have a promising result showing that directly learning from demonstration Q helps DDPG to converge faster.
    2. We have also shown that only providing perfect demonstrations are not enough.
    3. We need to show that whether the manually calculated Q value is similar, or ideally the same as the real Q value as the Q value given by a well trained critic. See experiments below.
    4. We also need to see that whether a trained demonstration neural network (ensemble) has enough/ or does not have enough uncertainty on (s, a) pairs it did not see.
        - (s,a) vs std dev on state and action pairs that were not seen in the training set.
        - fixed state then change action only, plot action vs q.
    1. Then, we should try to work with the (s, a) domain. We should, based on some critic, actively ask the expert for demonstrations for a particular $(s,a)$ pair.

## Topics for the meeting
### Experiments on Reach2DFO environment.
1. Environment Config
    - The goal of this environment is to go from anywhere in a 2D surface back to the origin (0,0). The starting point (x,y) satisfies $x, y \in Uniform(-1, 1)$
    - Input to the environment:
        - action $u \in R^2$ indicating the velocity (*not force*) for the next step (or simulation interval).
    - Output from the environment:
        - observation $o \in R^2$ indicating the current position
        - goal $g == (0,0)$ indicating the goal position.
        - reward $r \in R$
            - Suppose that the distance from current position to the goal position is $d$, the reward is either $r=-d$ or $r=1/(1+d)$
            - Note: for most of environments, I scaled down this reward by a factor of 12, and added a constant shift (0.5) to it (i.e. $r = (r+0.5)/12$).
        - flag: $is\_success$, set to true when the distance between current position and the goal position is smaller than a certain number (currently: 1/15).
    - Internal update rule for each step:
        - $o_{next}$ = $o_{curr} + 0.1u$
2. Experiments
    1. Manually calculated Q value is the same as output Q from a pre-trained critic? For each possible episode?
        - Train a ddpg environment used for generating output Q value. -> Modify parameters and get the result.
        - Use a new generate rollouts to generate episodes for each selected (s,a) pair as the starting point. -> Write a new generate rollout script to do so. Also modify the environment.
    3. Uncertainty check
        - This should be done in the demonstration training part.

## Meeting Minutes

## Reminder
