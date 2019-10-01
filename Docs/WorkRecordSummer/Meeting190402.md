# Undergrad Research Project
Meeting Summary<br>
April 2nd, 2019

## Goal/Thoughts/Idea from last week
    1. Instead of directly using $Q_R(s,a) = Q_D(s,a)$, try to use $Q_R(s,a) = r + \gamma Q_D(s',a')$
    2. Need to plot $Q_R(s,a)$, $Q_D(s,a)$, $\pi(s)$ and $V(s)$.
        For experiments below, after each epoch, I will check the above values by: <br>
            1. feeding state $s_{query}=(x, y)$ where $x,y=linspace(-1,1,32)$ to the ddpg actor and get its output $a_{actor}$<br>
            2. feeding state, action pair $(s_{query}, a_{actor})$ to the ddpg critic and demonstration neural net, which should give me $q_{rl}(s_{query}, a_{actor})$ and $q_d(s_{query}, a_{actort})$, respectively.<br>
        Since the actor in DDPG is deterministic, $q_{rl}$ and $q_d$ can also be considered as $v_{rl}(s_{query})$ and $v_d(s_{query}).$<br>
    3. Try to learn directly from a trained critic function, and compare the result with a demonstration neural net trained through supervised learning.

## Topics for the meeting
### Experiments Configuration
1. Environment Config
    - Using a new, simpler environment called "Reach2DFirstOrder".
    - The goal of this environment is to go from anywhere in a 2D surface back to the origin (0,0). The starting point (x,y) satisfies $x, y \in Uniform(-1, 1)$
    - Input to the environment:
        - action $u \in R^2$ indicating the velocity (*not force*) for the next step (or simulation interval).
    - Output from the environment:
        - observation $o \in R^2$ indicating the current position
        - goal $g == (0,0)$ indicating the goal position. Currently, this is always (0,0) because we want this environment to be as simple as possible.
        - reward $r \in R$
            - Suppose that the distance from current position to the goal position is $d$, the reward is either $r=-d$ or $r=1/(1+d)$
            - Note: for most of environments, I scaled down this reward by a factor of 12, and added a constant shift (0.5) to it (i.e. $r = (r+0.5)/12$). With this change, the ddpg critic converges to the true $q$ value faster and can learn from the demonstrator at the very beginning.
        - flag: $is\_success$, set to true when the distance between current position and the goal position is smaller than a certain number (currently: 1/15). In most of my experiments, I will also show a plot of success rate over epochs. The success rate is calculated based on this flag through multiple episodes.
    - Internal update rule for each step:
        - $o_{next}$ = $o_{curr} + 0.1u$
2. Experiment Variables
    1. Use either a trained ddpg critic as the Q output from demonstration $q_d$, or a neural net trained through supervised learning given lots of expert transition tuples $(s_t, a, s_{t+1}, q)$ where $q(s_t,a_t) = r_t + q(s_{t+1}, a_{t+1})$. The $q$ is manually calculated for each transition in each episode from the training set.
        - Input to the demonstration neural net: Observation $o \in R^2$ and action $u \in R^2$
        - Output from the demonstration neural net: Expected Q value $q_d$
        - Note: since we have a bootstrapped ensemble of demonstration neural nets, instead of having one $q_d$, we have $q_d^k$ where $k \in (1,n)$ and $n$ is number of ensembles. (Note: $n$ is set to 8 in all of my experiments.)
    2. How to use the demonstration Q value?
        - Suppose that we have a function $compare(q_r(s,a), q_d(s,a))$ that tells us what the desired $q(s,a)$ should be. Currently we have two updating rules:
        1. $q(s_t, a) = r + \gamma compare(q_r(s_{t+1},a'), q_d(s_{t+1},a'))$ where $a'$ is the output from ddpg actor given $s_{t+1}$
        2. $q(s_t, a) = compare(r + q_r(s_{t+1},a'), q_d(s_t,a))$
    3. What should $compare(q_{rl}, q_d)$ be?
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

## Reminder
- Generate a random data set of $(s,a)$, feed into the trained critic and get the output Q_demo, use $(s, a, Q_demo)$ to train the demonstration nn.
    - Reduce the size of the the data set gradually.
- if $q_{demo} > q_{rl}$
- 