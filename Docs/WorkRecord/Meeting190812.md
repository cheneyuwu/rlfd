# RL + IL Research Project
Meeting Summary\
Aug 12th, 2019


## Tasks & Planning

1. Compare the performance of MAF, RealNVP, and WGAN with gradient penalty (put in experiments together with the following)
2. Try generating the demo data with noise by changing the initial state and goal slightly
3. Try multiple starting positions without specifying the initial position (note: the goal should still be fixed)

4. In the mean time, you should also use the fetch reacher environment to further tuning the wgan

## Experiments

- OpenAIFetchPickAndPlace with fixed goal
    - Aug16FetchPickAndPlaceNFGridSearch
        - Need another grid search on maf and realnvp, as nf performs so bad.
        - Maybe consider reduce the capacity of the neural network?

- OpenAIFetchMove (2 objects) with fixed goal
    - Aug16FetchMove
        - Generate sub-optimal data with some gaussian noise.
        

## Issues

   
## Meeting Notes

1. setting network (policy q potential nets) initialization to zero (converges badly)
2. try different exploration mechanism (change the var of normal, change the )
3. try ensemble of discriminators and ensemble of policy separately


## Reminder

Aug 17 updated tf to 114 on cluster -> met seg fault again