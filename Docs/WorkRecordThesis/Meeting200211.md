# RL + IL Research Project
Meeting Summary\
Jan 18th, 2020


## Tasks & TODOs

- For learning from states
    - add prioritized experience replay? not for now
    - environments
        - old environments
            - on the old ddpg implementations it worked ok
        - metaworld environments
            - try to simplify the environment?
    - continue to tune the parameter for press_button with optimal demonstrations
    - try with sub-optimal demonstrations
    - for each environment you test with, check the generated potential

## Progress and Issue

### Noisy Demo Env
- TD3FDPegInsertion2D
    - OptDemo
        - Torch + BC
        - TF + MAF
            - check if this works, if it does not work then we should go back and check the old repo
    - Noise1.0
        - TD3 + BC
            - Works when setting prm loss to 1e-2 check if it works for 1e-4
- Metaworld
    - use only initialization, and RL+BC
    - simplify the environment

## Notes
- local
- rvl
- cedar
    - running with old tf code
- beluga
    - running with metaworld environments
        - press-button_topdown-v1
            - different parameters for nf shaping - no good parameter yet
            - running with increased time out - this is mostly done
        - press-button topdown v1 with sub optimal demonstrations
            - running with increased time out - mostly done
            - running with nf
- graham
    - running with metaworld environments
        - press-button
            - try running with the old tf code
        - press-button_topdown-v1 with noisy demonstrations 1.0
            - only bc for now -> **the result is too bad, reduce noise level**
            - running with nf
        - press-button_topdown-v1 with noisy demonstrations 0.5
            - running with increased time out
            - running with nf
        - press-button_topdown-v1 with noisy demonstrations 0.1
            - running with increased time out
            - running with nf

        - press-button_topdown-v1 with random init
            - actually we got similar result as single goal environment
            - but this is taking toooooooo long to train for maf

- check potential for both old environments **done**
    - actually looks similar than what we had before
- compare the result with suboptimal demonstrations and optimal demonstrations **done**
    - the one with sub-optimal demonstrations takes longer to converge? we should check this!
    - this is actually taking too long, run for longer!!!
- try reducing the training time for maf - less number of epochs works? **done**
    - trying with the press button env on beluga
- use the potential info to choose the parameter for maf **done**
    - we did not strictly follow the tuning process, but we still launched another run, let's check!
- also try tuning GAN **done**
    - running locally... again
- try using the old ddpg code for metaworld

## Reminder


- Run GAN on button press environments
    - measure sensitivity
    -
