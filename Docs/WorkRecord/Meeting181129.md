# Undergrad Research Project
Work Record<br>
December 1, 2018

## *Summary of last week*
- Read papers that were highlighted in our last meeting about exploration in RL, LfD and active learning. Also looked for other ideas.
- Set up simulators and played with some example environments.
- Looked into baseline implemntations of DDPG, DDPG+HER and DDPG+HER+Demo by OpenAI
- Went through experiments and result analysis from previous papers and though about evaluation and criteria of our task.

## Paper Highlight
These are papers we considered important or should be relevant to our tasks/goals. I found come baseline implementations of these methods and tried to start with them for our tasks.
- [DDPG with Demonstrations](https://arxiv.org/abs/1707.08817)
  - Early paper that added demonstration data into replay buffer. Demonstrations were kept forever.
  - The key ideas:
    - Prioritized replay to enable efficient propagation of the reward information
    - Mix of 1-step and n-step returns when updating the critic function.
  - The above 2 ideas were well explained. Besides those, they also mentioned
    - Learning multiple times per environment step (not very clear to me)
    - Regularization
- [DDPG+HER](https://arxiv.org/abs/1707.01495)
  - OpenAI method for solving robotics tasks such as FetchPickPlace. It introduced the notion of adding "goal" into reward function and learning from failures. Last time we discussed about tasks with random initial state and goal state, which seems to be exactly what this paper was trying to do.
- [DDPG+HER+Demonstration](https://arxiv.org/pdf/1709.10089.pdf)
  - This is the method that I want to start with. It combines the previous two papers and makes further improvement such as
    - MSE loss for Demonstraion examples to train the actor
    - Q-filter to overcome the problem that demonstrations can be suboptimal
  - This method has been implemented on the robotics environments of OpenAI gym. It is  used for solving similar problems as PegInHole. I think I can implement this idea on our task and then try to improve it with ideas of exploration discussed last week.
- [Ensemble of Q functions](https://arxiv.org/abs/1602.04621)
  - This was one of the main idea we discussed last time. The idea paper is very clear and makes sense. I was also looking for papers that might be interesting from its citations.
  - I want to try this idea first once I have my basic DDPG implementation ready for our task.
- [VIME](https://arxiv.org/abs/1605.09674)
  - The overall idea/method is clear. Read and understood their math and derivations, some derivation still seems a little confusing.
- [Asymmetric Self-Play](https://openreview.net/forum?id=SkT5Yg-RZ)
  - Raper mentioned in the last email that introduced a new method of exploration. The general idea is having one agent (A) to produce random goals for the other agent (B) to complete. Agents learn about the environment throughout this iterative process.

## Simulators
- Set up Bullet and Mujoco on my home machine
- Robosuite
  - I am interested in this repo. It provides a version of PegInHole and some similar tasks. It also provides a simple way of creating new tasks. (I am still looking into it.) Use Mujoco.
- Roboschool
  - The examples envs here are mostly run/walk tasks. Use Bullet.
- OpenAI Gym
  - Played with some of their Robotics Environments. (Not exactly PegInHole but similar tasks.)
  - Looked into their DDPG+HER+Demo baseline implementations on these environments.

## Implementations
- Looked into OpenAI implementation of DDPG and DDPG+HER. Trying to implement it on new tasks.
- Also exploring this repo [Surreal RL Framework](https://github.com/SurrealAI/surreal). If we use Robosuite, some baseline implementations here should be helpful.

*TODO*
- One thing we did not discuss much last time was model based method. There are quite a few papers combining above ideas with model based methods, which I am currently reading.
- [OpenAI Robotics Research](https://blog.openai.com/ingredients-for-robotics-research/) also has a list of possible improvements to HER. I think some of them might be related to our idea (e.g. unbiased HER or HER+HRL).

## Reminder 
- Account for compute canada cluster
