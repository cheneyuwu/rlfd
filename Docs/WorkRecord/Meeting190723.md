# RL + IL Research Project
Meeting Summary\
July 23th, 2019


## Tasks & Planning
- Prove that maf does not add noise to the rl training, and can be helpful
- Is it possible that we use a policy trained through imitation learning as the potential function and use rl to shape this reward function?
    - The trained potential function does not give correct value at states not seen in the demonstration data set.

## Experiments
- Experiments:
    - OpenAI Fetch Reacher with fixed goal
        - (July 23) Run with bc, rb, rbmaf, maf and pure rl. maf ran with 500 loss ratio and weight equals to 5. Looks like maf is completely adding noise to the trainig process.
        - (July 24) Ran a grid search on the hyper parameters for the maf potential. Found that weight ratio 1000 and weight equals to 3 actually works well on this single goal environment. However, it is not sure whether these parameters can easily be tuned on more complicated environments. (Tried this parameter on the multigoal environments yesterday).
            - Additionally, it also takes extremely long time to learn this behavior.

## Issues
- Whether or not the learnt potential function is correct, desired.
- Whether or not the critic function compensates the potential function at the beginning?
   
## Meeting Notes
- Hyper parameter search for pick and place
    - Noisy, ~30% success rate
- smaller number of demonstrations | more noise in the demonstrations
- more random seeds
- project to see the smoothness of the potential function

- rerun the test without training the critic function on demonstration data (i.e. do not assume that the demonstration data has rewards.)
## Reminder