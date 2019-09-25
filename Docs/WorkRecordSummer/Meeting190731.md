# RL + IL Research Project
Meeting Summary\
July 31st, 2019


## Tasks & Planning

- Assume that the demonstration does not contain reward. So we can only add them to the policy training.

1. Tune the hyper-parameters for behavior cloning (aux_weight_loss) (July 29)
2. For single goal pick and place, add more noise to the demonstration (July 29)
3. Use the replay buffer that stops once it reaches the goal
    - Note that you need to rerun the BC version for this
4. Try increasing the capacity of the MAF and see if this helps (Maybe consider reducing the capacity of the network)
5. Consider to use W-GAN instead of MAF
6. Consider using other type of normalizing flows such as RealNVP
7. Try multigoal fetch pick and place

## Experiments
- Experiments:
    - OpenAIFetchPickAndPlace with fixed goal
        - (July 29) FetchPickAndPlaceBCGridSearch
        - (July 29) FetchPickAndPlaceDemoNoise
            - Add noises with different mean and variance to the demo data, compare the performance of the rl agent
        - (July 31) FetchPickAndPlaceDemoNoise[BC, MAFEns4, MAFEns8]
            - Add noises with different mean and variance to the demo data, compare the performance of the rl agent
            - run with multiple seeds
        - (Aug 1) FetchPickAndPlaceMAFGridSearch(_2)
            - Use this experiment to tune your parameter for maf flow, we should focus on the weight for the regularizer for now
            - Probabily just use an ensemble of 1 for now for performance
        - (Aug 1) FetchPickAndPlaceMAFCapacitySearch
            - Quick check what happens if you increase the capacity of the normalizing flow (this is probabily just another parameter tuning issue)
        - (Aug 1) FetchPickAndPlaceBCNonFixedT
            - Check the performance of bc after using the non-fixed episode length, also compare with maf
        - (Aug 1) FetchPickAndPlaceMAFNonFixedT
            - Similar to the above experiment. -> make sure that you do this after the grid search. otherwise the result cannot be promising.
            - After this experiment, we should consider to switch to the non-fixed version
    - OpenAIFetchPickAndPlace with multigoal
        - () FetchPickAndPlaceBCGridSearch
        - () FetchPickAndPlaceMAFGridSearch

## Issues
   
## Meeting Notes
- smaller number of demonstrations | more noise in the demonstrations

## Reminder