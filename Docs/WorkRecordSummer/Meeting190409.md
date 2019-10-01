# Undergrad Research Project
Meeting Summary<br>
April 9th, 2019

## Goal/Thoughts/Idea from last week
    1. We now have a promising result showing that directly learning from demonstration Q helps DDPG to converge faster. However, we also noted that probably only providing perfect demonstrations are not enough. We need more experiments to show that whether it is sufficient to use only perfect demonstration or not. See experiments below.

## Topics for the meeting
### Experiments Configuration
1. Environment Config
    - Use both the "Reach2DFirstOrder" and the "Reach2D" environment. Refer to the notes form last meeting for details about these two environment.
2. Experiment Variables
    1. How should we generate the training set?
       1. Use only expert episodes as the training set.
       2. Use randomly generated episodes as the training set.
       3. Use pre-selected state and action pairs as the training set.
    2. Use either a trained ddpg critic as the Q output from demonstration $q_d$, or a neural net trained through supervised learning given lots of expert transition tuples $(s_t, a, s_{t+1}, q)$ where $q(s_t,a_t) = r_t + q(s_{t+1}, a_{t+1})$. The $q$ is manually calculated for each transition in each episode from the training set.
        - Input to the demonstration neural net: Observation $o \in R^2$ and action $u \in R^2$
        - Output from the demonstration neural net: Expected Q value $q_d$
        - Note: since we have a bootstrapped ensemble of demonstration neural nets, instead of having one $q_d$, we have $q_d^k$ where $k \in (1,n)$ and $n$ is number of ensembles. (Note: $n$ is set to 8 in all of my experiments.)
    3. How to use the demonstration Q value?
        - Suppose that we have a function $compare(q_r(s,a), q_d(s,a))$ that tells us what the desired $q(s,a)$ should be. Currently we have two updating rules:
        1. $q(s_t, a) = r + \gamma compare(q_r(s_{t+1},a'), q_d(s_{t+1},a'))$ where $a'$ is the output from ddpg actor given $s_{t+1}$
        2. $q(s_t, a) = compare(r + q_r(s_{t+1},a'), q_d(s_t,a))$
    4. What should $compare(q_{rl}, q_d)$ be?
        - Currently we have three ways to compare $q_r$ and $q_d$. Using bootstrapped ensemble of demonstration neural nets, we have $q_d^k$ where $k \in (1,n)$ and $n$ is number of demonstraion neural network ensembles.
        1. Compare $q_r(s,a)$ with the mean value of $q_d(s,a)$ minus variance of $q_d(s,a)$.
            - $compare(q_r(s,a), q_d(s,a)) = max(q_r(s,a), mean(q_d(s,a)) - var(q_d(s,a))$
        2. Count how many outputs $q_d^k$ are greater than $q_r$
            - $c = count(q_r(s,a) > q_d^k(s,a))$ where $k \in (1,n)$
            - if $c > 0$ then use $q_r(s,a)$ otherwise use $mean(q_d(s,a))$
                - Currently, we choose $mean(q_d(s,a))$ only when the outputs from all demonstration neural net ensembles are greater than $q_r$.
        3. Use a weighted combination of $q_d(s,a)$ and q_r(s,a)$
            - $c = count(q_r(s,a) > q_d^k(s,a))$ where $k \in (1,n)$ -> same as method 2
            - $w$ = $c/n$
            - $q(s,a) = w*q_r(s,a) + (1-w)*q_d(s,a)$ -> but instead of having a hard threshold, we use a weighted combination.

## Meeting Minutes
1. Preselect (s,a) that covers the entire state action space then for each (s,a), generate a full episode using this (s,a) pair as the start, calculate the corresponding q.
Check the difference b/t pre-trained critic q and computed expert q.
2. Try to work with the (s,a) domain
    actively asking the expert for demonstration for particular s a pairs.
3. bootstrapping (match true variance) vs dropout baysian
    - state, action vs std dev (give (s,a) pairs that do not cover the entire state action space.)
    - fixed state then change action only, plot action vs q.
## Reminder