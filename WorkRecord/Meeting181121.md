# Undergrad Research Project
Meeting Summary<br>
November 21, 2018

## Project Idea/Definition/Goal
- Main Ideas
  - Adding uncertainties to the Q function, which can be done by either using ensembles of Q functions or using dropout layers.
  - Think about if we can combine active learning of goals and imitation learning. Or, how can they be part of the same system.
    - Both methods can help train the Q function. Imitation learning provides a sample of optimal trajectory and corresponding rewards $\{r^1,...,r^n\}$, which can be used to learning the Q function using supervied learning. Active learning reduces model uncertainty.
    - *TODO:* Look into papers about this topic suggested in the email and online.
- Other Ideas
  - Assume the program can ask expert for demonstrations, when should it ask?
    - We can have limited number of queries allowed, and try to do you best.
  - (Side Topic) Automatic goal generation (David Held)
    - [Reverse Curriculum Generation for Reinforcement Learning](https://arxiv.org/pdf/1707.05300.pdf)
- Model based DDPG
  - [Model-based DDPG for motor control](https://ieeexplore.ieee.org/document/8359558)
    - This paper might be interesting as it uses model based DDPG for motor control. The task their program learnt also had random initial and final states.

## Example Tasks
- Peg in a hole
  - Can have multiple holes whose relative position is fixed.
  - Random initial state + random goal position + user specified target hole
  - *TODO:* Try to find a good simulator for this.

- Navigation
  - Deepmind Lab Maze
  - Google street view navigation simulator (Piotr Mirowski and Raia Hadsell)
  - Program inputs are images. Human can get the 2D overview of the entire map and provide trajectory demonstrations.
  - 

- Other problems
  - *TODO:* For each task, think about:
    - How to evaluate our program? Criteria?
    - What experiments should be done?
    - For peg in a hole, we can find relevant papers and see criteria they use.

## Reminder 
- Account for compute canada cluster
