# Shaping Rewards for Combined Reinforcement and Imitation Learning using Generative Models 
- by Yuchen Wu, Melissa Mozifian and Prof. Florian Shkurti

Note: This codebase has not been cleaned up yet. Will do it soon!

## [Link to Paper](.)

## Installation
1. Setup Mujoco and install mujoco_py
2. Clone this repo and its `gym` submodule
    - `git clone git@github.com:cheneyuwu/RLProject`
    - `git submodule init`
    - `git submodule sync`
    - `git submodule update --remote`
3. Install packages at `./Package/gym` and  `./Package/td3fd`
4. Source the setup script at root directory, this script just defines some environment variables for the logger.
    - `source setup.py`

## Examples of running experiments
- Create a folder to store your training result ```mkdir <folder name> && cd <folder name>```
- Inside this folder, create a python file storing the parameters as a dictionary you want to use ```touch parameters.py```
    - The parameter file should look like ones in `./Package/td3fd/td3fd/ddpg/param/*`
    - Optionally, in the python file you just created, you can have something like:
        ```
            from copy import deepcopy
            from td3fd.ddpg.param.ddpg_fetch_peg_in_hole_2d import params_config as base_params
            
            params_config_gan = deepcopy(base_params)
            params_config_gan["ddpg"]["demo_strategy"] = ("gan", "none")
            params_config_gan["seed"] = tuple(range(10))   
        ```
        - Note 1: the name of the variable should start with `params_config_`
        - Note 2: if a parameter is of type `tuple`, it is assumed that you want to iterate over the tuple and train multiple agents each with a different parameter listed in the `tuple`
        - In the above script, we imported the default parameters ued for the 2D peg in hole environment, then we override some of them parameters:
            - `params_config_gan["ddpg"]["demo_strategy"] = ("gan", "none")` means to train two agents, one with GAN shaping, one with just TD3
            - `params_config_gan["seed"] = tuple(range(10))` means to train 10 agents with seed 0 to 10
        - Therefore, this modified parameter dictionary tries to run 20 experiments, TD3 + GAN shaping with seed 0-10 and TD3 with seed 0-10.
        - See `./Package/td3fd/td3fd/ddpg/config` for a detailed description of all parameters
- Also in this folder, you should have your demonstration data (not needed if you just want to run TD3), `demo_data.npz`
    - TODO: explain the format of demonstration data.
- Now you can launch the training process, using the parameters defined in `parameters.py` through the following command:
    ```mpirun -n 20 python -m td3fd.launch --targets train:parameters.py```
    - If you would like to start looking into the whold td3fd package, start with the `./Package/td3fd/launch.py` script, which is used to launch everything.
- You can stop training by `Ctrl-C` and restart the process using exactly the same command at any time.

## TODO