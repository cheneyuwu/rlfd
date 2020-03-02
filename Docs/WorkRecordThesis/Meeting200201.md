# RL + IL Research Project
Meeting Summary\
Jan 18th, 2020


## Tasks & TODOs

- For learning from states
    - **add n step return to the replay buffer**
    - environments
        - meta world
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

### Noisy Demo Env
- TD3FDPegInsertion2D
    - OptDemo
        - Torch + BC
        - TF + MAF
            - check if this works, if it does not work then we should go back and check the old repo
    - Noise1.0
        - TD3 + BC
            - Works when setting prm loss to 1e-2 check if it works for 1e-4
        -
- RlkitPegInsertion2D
    - OptDemo
    - Noise1.0

## Notes

## Reminder
