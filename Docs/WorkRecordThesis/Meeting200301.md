# RL + IL Research Project
Meeting Summary\
Mar 01st, 2020


## Current Tasks / Progress

- add prioritized experience replay? not for now
- metaworld environments - try to simplify the environment?
- continue to tune the parameter for press_button with optimal demonstrations
    - The problem is that with some parameters, the solutions is not optimal
    - It might be worth to find the trade-off?, or at least minimize the effect of sub-optimality?

## Problems / Issues
- The problem is that with some parameters, the solutions is not optimal

## Experiments / Results.

- Note: for all results below, we run with multi-step return by default
- Note: also we initialize with bc whenever possible

- GYM
    - Peg In Hole
        - RL Only
        - BC Only
        - BC Initialization
        - RL + IL & RL + IL with Q-Filter
        - GAN Shaping
        - NF Shaping
    - Pick and Place
        - **RL Only**
        - **BC Only**
        - **BC Initialization**
        - RL + IL
        - RL + IL with Q-Filter
        - GAN Shaping
        - NF Shaping

- MetaWorld
    - press-button_topdown-v1 with optimal-demonstrations (beluga)
        - BC Only
        - RL Only
        - BC Initialization
        - RL + IL & RL + IL with Q-Filter
        - GAN shaping
        - GAN shaping with weight decay
        - NF shaping with weight decay
        - NF shaping

    - press-button topdown v1 with sub-optimal demonstrations (beluga)
        - BC Only
        - RL Only
        - BC Initialization
        - RL + IL & RL + IL with Q-Filter
        - GAN shaping
        - NF shaping
        - GAN shaping with weight decay
        - NF shaping with weight decay

    - press-button topdown v1 with noisy demonstrations - 0.1 (graham)
        - BC Only
        - RL Only
        - BC Initialization
        - RL + IL & RL + IL with Q-Filter
        - GAN shaping
        - NF shaping
        - GAN shaping with weight decay
        - NF shaping with weight decay

    - press-button topdown v1 with noisy demonstrations - 0.5 (graham)
        - BC Only
        - RL Only
        - BC Initialization
        - RL + IL
        - RL + IL with Q-Filter
        - GAN shaping
        - NF shaping
        - GAN shaping with weight decay
        - NF shaping with weight decay

    - press-button_topdown-v1 with random initial pose (graham)

## Meeting Notes
- try with meta gradient rl!
- try with support and the threshold method
    - check the two papers sent!!!
- try tune the exploration parameter -> time dependent noise
    - ask Scott about this!

## Reminder

## Other Notes