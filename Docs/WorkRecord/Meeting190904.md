# RL + IL Research Project
Meeting Summary\
Aug 27th, 2019


## Tasks & Planning

1. Compare the performance of MAF, RealNVP, and WGAN with gradient penalty (put in experiments together with the following)
2. Try multiple starting positions without specifying the initial position (note: the goal should still be fixed)



## Issues

- TODO list in the code:
    1. robot_env line 73 and 90, comment out to generate demonstration data.
    2. launch.py line 90 excluded params for plotting


## Meeting Notes


## Reminder

## Experiments (Final Experiments)

- Run pure bc, rl + bc with q filter, rl + bc without q filter, rl + gan, rl + maf
- Generate suboptimal trajectories, making sure that you have 3 type of noise in the demonstration data (For the first sets, run 4)
- Run Experiments with PegInHole, PegInHoleRandInit, PickAndPlace

- Noise types:
    - Sub-Optimal Path with 2 levels: 1, 2  (S)
    - Variance in demo with 2 levels: <var>  (V)
    - Gaussian Noise with 2 levels: zero and some  (G)


- Peg In Hole Rand Init
    - S0.0V0.02G0.0
        - no_shaping **running on beluga-melissa** (done)
        - gan        **running on florian's lab** (done)
            - with weight = 3.0?
        - maf        **running on florian's lab** (done)
        - pure_bc    **running on florian's lab** (done)

    - S0.1V0.02G0.0
        - no_shaping **running on beluga-melissa** (done)
        - gan        **running on graham-melissa** (done)
        - maf        **running on graham-melissa** (done)
        - pure_bc    **running on florian's lab** (done)

    - We probabily need a larger noise level for this run? (The current looks good enough)

- Pick And Place
    - S0.0V0.0G0.0
        - no_shaping **running on beluga-melissa** (done)
        - gan        **running on beluga-melissa** (done)
        - maf        **running on graham-melissa** (done)
        - pure_bc    **running on florian's lab** (done)

    - S0.1V0.1G0.0 (Not Using)
        - no_shaping **running on beluga-melissa** (done)
        - gan        **running on beluga-melissa** (done)
        - maf        **running on graham-florian** (running)
        - pure_bc    **running on florian's lab** (running)

    - S0.3V0.2G0.0
        - no_shaping **running on beluga-melissa** (done)
        - gan        **running on beluga-melissa** (done)
        - maf        **running on graham-melissa** (running)
        - pure_bc    **running on florian's lab** (done)

    - S0.3V0.2G0.0Tree
        - no_shaping **running on graham-yuchen** (done)
        - gan        **running on graham-yuchen** (done)
        - maf        **running on graham-melissa** (running)
        - pure_bc    **running on florian's lab** (done)

- Pick And Place Rand Init
    - S0.0V0.1G0.0long
        - no_shaping **running on graham-melissa** (done)
        - gan        **running on graham-melissa** (done)
        - maf        **running on graham-yuchen** (running)
        - pure_bc
    - S0.3V0.2G0.0long
        - no_shaping **running on graham-melissa** (done)
        - gan        **running on graham-melissa** (done)
        - maf        **running on graham-yuchen** (running)
        - pure_bc




# New Experiments

- Peg In Hole 2D Version
    - Robust To Noise (4 seeds for each)
        - Extra Noise 0.0
            - RLBC 0.0001 0.001 0.01 0.1 **running on florian's lab** (done)
            - MAF **running on florian's lab** (done)
            - GAN **running on florian's lab** (done)
        - Extra Noise 0.2
            - RLBC 0.0001 0.001 0.01 0.1 **running on florian's lab** (done)
            - RLBCQ 0.0001 **running on florian's lab** (done)
                - need 0.1
            - GAN **running on florian's lab** (done)
            - MAF
        - Extra Noise 0.5
            - RLBC 0.0001 0.001 0.01 0.1 **running on florian's lab** (done)
            - RLBCQ 0.0001 **running on florian's lab** (done)
                - need 0.1
            - GAN **running on florian's lab** (done)
            - MAF        
        - Extra Noise 1.0
            - RLBC 0.0001 0.001 0.01 0.1 **running on florian's lab** (done)
            - RLBCQ 0.0001 **running on florian's lab** (done)
                - need 0.1
            - GAN **running on florian's lab** (done)
            - MAF **running on florian's lab** (done)
    - RLBCWeight (Use Noise 1.0) BC loss : RL loss
        - 10000 : 1
        - 1000 : 1
        - 100 : 1
        - 10 : 1

- Peg In Hole with Randome Initial Pose (Robust to Seeds)
    - S0.0 (Near Optimal Policy)
        - rl **running on cedar melissa** (done)
        - bc **running on cedar melissa** (done)
        - rlbc **running on cedar melissa** (done)
        - gan **running on cedar melissa** (done)
        - maf **running on cedar yuchen**  (done)
            - (true ensemble) running with more n epochs and tunning **running on cedar melissa** (running)
            - (true ensemble) running with tunning **running on cedar yuchen** (done)
    - #S0.15 (Sub Optimal Policy: goes high and then insert)

- Pick and Place with Random Initial Pose (Robust to Seeds)
    - S0.0 (Near Optimal Policy)
        - rl **running on cedar florian** (done)
        - bc **running on cedar florian** (done)
        - rlbc **running on cedar florian** (done)
        - gan **running on cedar florian**
            - (true ensemble) more tunning and more epochs **running on graham melissa** (running)
            - (true ensemble) more tunning and more epochs **running on graham florian** (running)
        - maf **running on cedar yuchen**
            - (true ensemble) running with more n epochs and tunning **running on cedar florian** (running)
            - (true ensemble) running with tunning **running on cedar yuchen** (done)
            - (true ensemble) try increasing the learning rate **running locally** (done)

- **TODO**
    - MAF for PegInHole2DVersion (0.2 and 0.5) **running on graham yuchen** (done)
    - Q filter for Robust to success rate **running on graham florian** (done)


# Lunch and Uber cost
    - Aug 18 Sushi lunch: (13.95 + 5.49[delivery fee] / 2 ) * 1.13[tax] = 18.86
    - Aug 24 Chicken lunch: 49.57 / 3 = 16.52
    - Aug 24 Uber with Melissa from DT to UTM: 31.66
    - Aug 30 Uber to lunch place: 8.55
    - Aug 31 Sushi lunch: 57.95 / 3 = 19.31
    - Total: 18.86 + 16.52 + 31.66 + 8.55 + 19.31 = 94.89