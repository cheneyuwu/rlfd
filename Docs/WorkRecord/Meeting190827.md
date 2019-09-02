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

- Peg In Hole

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
        - maf        **running on graham-melissa** (running)
        - pure_bc    **running on florian's lab** (running)

    - S0.1V0.1G0.0
        - no_shaping **running on beluga-melissa** (done)
        - gan        **running on beluga-melissa** (done)
        - maf        **running on graham-florian** (running?)
        - pure_bc    **running on florian's lab** (running)

    - S0.3V0.2G0.0
        - no_shaping **running on beluga-melissa** (done)
        - gan        **running on beluga-melissa** (done)
        - maf        **running on graham-melissa** (running)
        - pure_bc    **running on florian's lab** (running)

    - S0.3V0.2G0.0Tree
        - no_shaping **running on graham-yuchen** (done)
        - gan        **running on graham-yuchen** (done)
        - maf        **running on graham-melissa** (running)
        - pure_bc    **running on florian's lab** (running)

- Pick And Place Rand Init
    - S0.0V0.1G0.0long
        - no_shaping **running on graham-melissa** (running)
        - gan        **running on graham-melissa** (running)
        - maf
        - pure_bc
    - S0.3V0.2G0.0long
        - no_shaping **running on graham-melissa** (running)
        - gan        **running on graham-melissa** (running)
        - maf
        - pure_bc

# Lunch and Uber cost
    - Aug 18 Sushi lunch: (13.95 + 5.49[delivery fee] / 2 ) * 1.13[tax] = 18.86
    - Aug 24 Chicken lunch: 49.57 / 3 = 16.52
    - Aug 24 Uber with Melissa from DT to UTM: 31.66
    - Aug 30 Uber to lunch place: 8.55
    - Aug 31 Sushi lunch: 57.95 / 3 = 19.31
    - Total: 18.86 + 16.52 + 31.66 + 8.55 + 19.31 = 94.89