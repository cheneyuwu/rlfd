# RL + IL Research Project
Meeting Summary\
Aug 7th, 2019


## Tasks & Planning

1. Compare the performance of MAF, RealNVP, and WGAN with gradient penalty (put in experiments together with the following)
2. Try generating the demo data with noise by changing the initial state and goal slightly
3. Try multiple starting positions without specifying the initial position (note: the goal should still be fixed)

4. In the mean time, you should also use the fetch reacher environment to further tuning the wgan

## Experiments

- OpenAIFetchPickAndPlace with fixed goal
    - Compare the effect of using fixed t and non fixed t
        - **Aug8FetchPickAndPlaceBCNonFixedT**
        - **Aug8FetchPickAndPlaceBCFixedT**
        - **Aug8FetchPickAndPlaceMAFNonFixedT**
        - **Aug8FetchPickAndPlaceMAFFixedT**
    - Generating the demo data with noise by changing the initial state and goal slightly
        - **Aug9FetchPickAndPlaceWithDemoNoise**
            - adding trajectory level noise to the demo data
            - this experiment should be run with fixed T, and using gan maf realnvp and bc
        - **Aug9FetchPickAndPlaceWithoutDemoNoise**
            - use this as a direct compare with the one above

## Issues

1. Try increasing the capacity of the MAF and see if this helps
    1. I was wrong about the number of maf bijectors used in our normalizing flow. It was 6 instead of 3.
    2. Played with learning rate. Noticed that we achieved similar performance even if we stop the maf training earlier. Previously I did this early stop by mistake because the learning rate was so slow such that the loss dropped so slow. I thought that it converged already but actually it did not. If I use a larger learning rate, the loss dropped significantly. However, the question is, if we can achieve similar performance by stopping earlier, does it mean that the nf have enough capacity already? (Maybe consider reducing the capacity of the network)
2. After I started using env that terminates once the goal is reached. The performance of bc did not drop very much, however, the performance of maf dropped significantly. 
    1. Need more parameter tuning for this?
3. Consider to use W-GAN instead of MAF
    1. Need to test with simple, toy environments first.
4. Consider using other type of normalizing flows such as RealNVP
   
## Meeting Notes


## Reminder