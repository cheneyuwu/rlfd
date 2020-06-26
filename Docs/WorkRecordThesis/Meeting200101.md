# RL + IL Research Project
Meeting Summary\
Jan 01st, 2020


## Tasks & TODOs

- For learning from states
    - environments
        - meta world -> see above
            - need to tune parameters!
        - GYM mujoco environments
            - this works with just TD3 need to make the GAN part work later
        - GYM robotics and your own environments! make them work with the pytorch implementation

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

## Progress and Issues

- syncing tf and torch implementation of the algorithm
    - do not forget that you have added a switch between the two algorithms!
- run experiments on server

- TODOs:
    - experiments with meta world
    - make sure that GAN potential with normalizer works on the torch implementation
    - add ensemble to torch implementation
    - make MAF working, ideally with torch, but currently we do not have that much time, just focus on GAN
    - also do not forget GAIL comparison, although given the current timeline, cleaning this up will be hard. Also we need time to tune the parameters, we surely don't have that much time

## Meeting Notes

## Thesis Notes

## TODOs
- D4PG, A3C algorithms on image based environments.

## Related Works
- PlaNet (Learning Latent Dynamics for Planning from Pixels)
    - environments:
        - DeepMind Control Suite, input image is 64 * 64 * 3
    - main ideas
        - planning in latent spaces
            - model based planning: model predictive control
        - recurrent latent space model
        - latent overshooting

## Reminder
