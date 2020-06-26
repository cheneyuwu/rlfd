# RL + IL Research Project
Meeting Summary\
Dec 30th, 2019


## Tasks & TODOs

## Progress & Issues

## Meeting Notes

- For learning from images
    - input:
        - input is set to 84 x 83 x 3 (according to Melissa)
        - mount the camera on the arm
        - back ground changes (for GAN learning)
    - modifications to GAN
        - replicate actions to make the number of dims of action and observation close
    - modifications to TD3
        - CNN 
        - frame stacking
        - sticky actions
    - algorithms to consider other than TD3
        - model ensemble TRPO/MPPI
    - algorithms to compare
        - PlaNet, D4PG (do not compare to it for now), SAC+AE, SLAC
    - environments
        - meta world -> it has some meta learning baselines, but for single environment, check the sister project "garage"
            - does not have baselines for image based environments, need the trick
        - deepmind suite -> some simple environments and image based baselines (PlaNet)
- For learning from states
    - environments
        - meta world -> see above
            - need to tune parameters!
        - GYM mujoco environments
            - this works with just TD3 need to make the GAN part work later
        - GYM robotics and your own environments! make them work with the pytorch implementation

## Reminder

