# Shaping Rewards for Combined Reinforcement and Imitation Learning using Generative Models
- by Yuchen Wu, Melissa Mozifian and Prof. Florian Shkurti

## [Link to Paper](.)

## Prerequisites
1. This repository requires Python3 (>=3.5) and is tested with python 3.6.
2. You will also need [MuJoCo](http://mujoco.org/) & [mujoco_py](https://github.com/openai/mujoco-py) to run experiments for the PegInsertion and PickAndPlace environments as we did for the paper. However, we also provide a toy 2D reacher environment for quick testing and tutorial purpose, so MuJoCo is optional if you are not interested in reproducing the results.
3. We also support training multiple models (with different parameters) in parallel through [OpenMPI](https://www.open-mpi.org/) & [mpi4py](https://mpi4py.readthedocs.io/en/stable/) but is also optional.


## Installation
1. Clone this repo and its `gym` submodule including the customized PegInsertion and PickAndPlace environments
    ```
    git clone git@github.com:cheneyuwu/TD3fD-through-Shaping-using-Generative-Models
    git submodule init
    git submodule sync
    git submodule update --remote
    ```
2. Install packages at `<root>/Package/gym` and  `<root>/Package/td3fd`
    ```
    pip install -e <root>/Package/gym
    pip install -e <root>/Package/td3fd
    ```
    In terms of Tensorflow, `td3dfd` requires `tensorflow==1.15.0` and `tensorflow_probability==0.7.0`. \
3. Source the setup script at root directory, this script just defines some environment variables for logging.
    ```
    source <root>/setup.py
    ```

## Running Experiments

### Generating Demonstrations
Demonstration data must be stored as a `npz` file named `demo_data.npz`, it should contain the following entries:
- "o": array of observations of shape (number of episodes, episode_length+1, observation dimension)
- "u": array of actions of shape (number of episodes, episode_length, action dimension)
- "g": (optional) array of goals of shape (number of episodes, episode_length, goal dimension)
- "ag": (optional) array of achieved goals of shape (number of episodes, episode_length+1, goal dimension)
- "r": observations of shape (number of episodes, episode_length, 1)

For our experiments, we provide two options for generating demonstration data.
1. Using hard coded scripts. Refer to `<root>/Experiment/<envrionment names>/README.md` for more information.
2. Using pre-trained scripts. Our implementation of DDPG/GAIL stores models through pickle as `<policy name>.pkl`. To use a pre-trained model to generate demonstration data:
    1. create a json file named `demo_config.json` containing the following:
        ```
        {
            "seed": 0,                  // Seed environment
            "num_eps": 40,              // Number of episodes to be generated
            "fix_T": true,              // Episode length is fixed
            "max_concurrency": 10,      // Maximum number of environments running in parallel
            "demo": {
                "random_eps": 0.0,      // Chance of selecting a random action
                "noise_eps": 0.1,       // Variance of gaussian noise added to action, scaled by maximum action magnitude
                "render": false         // Rendering (I suggest using max_concurrency = 1 when setting this to True)
            },
            "filename": "demo_data.npz" // Output file name. This is the default name consumed by the training scripts.
        }
        ```
    2. put the pre-trained `<policy name>.pkl` in the same folder as your `demo_config.json` and then run
        ```
        python -m td3fd.launch --targets demo_data --policy_file <policy name>.pkl
        ```

### Training Models
- Create a python script storing parameters to be used for training your model, named `<param name>.py`
    - We have pre-defined parameter files for the environments we tested in `<root>/Package/td3fd/td3fd/ddpg/param/*` and
    - Optionally, in the python file you just created, you can simple import from our pre-defined parameters, e.g.
        ```
        from copy import deepcopy
        from td3fd.ddpg.param.ddpg_fetch_peg_in_hole_2d import params_config as base_params

        params_config = deepcopy(base_params)
        params_config["ddpg"]["demo_strategy"] = ("gan", "none")
        params_config["seed"] = tuple(range(10))
        ```
    Note 1: the name of the global variable starts with `params_config`. \
    Note 2: if a parameter is of type `tuple`, it is assumed that you want to iterate over the tuple and train multiple agents each with a different parameter listed in the `tuple`. \
    In the above script, we imported the default parameters used for the 2D Peg Insertion environment, then we override some parameters:
    - `params_config_gan["ddpg"]["demo_strategy"] = ("gan", "none")` means to train two agents, one with reward shaping via GAN, one with just TD3 (as `use_TD3` has been set `True` in `ddpg_fetch_peg_in_hole_2d`)
    - `params_config_gan["seed"] = tuple(range(10))` means to train 10 agents with seed 0 to 10.

    Therefore, this modified parameter dictionary tries to run 20 experiments, TD3 + GAN shaping with seed 0-10 and TD3 with seed 0-10. \
    See `<root>/Package/td3fd/td3fd/ddpg/config` and `<root>/Package/td3fd/td3fd/gail/config` for a detailed description of all parameters
- Put the `demo_data.npz` generated in previous step in the same folder as `<param name>.py`
    - Note: If you only want to run our implementation of TD3, then `demo_data.npz` is not needed.
- In the same folder, run
    ```
    (mpirun -n 20) python -m td3fd.launch --targets train:<param name>.py
    ```
    Note: Use `mpirun -n 20` if you have OpenMPI installed and want to run all 20 experiments in parallel.
- After training, the following files should present in the each experiment directory:
    ```
    demo_data.npz - demontration data, which is copies to each run
    log.txt       - training logs
    params.json   - parameters for the experiment enclosed in this directory
    progress.csv  - experiment statistics for plotting
    policies      - a folder containing intermediate policies, <policy_[0-9]+>.pkl, and the last one, policy_last.pkl
    ```

### Evaluating/Visualizing Models
```
python -m td3fd.launch --targets evaluate --policy_file <policy file name>.pkl
```

### Plotting
```
python -m td3fd.launch --targets plot --exp_dir <top level plotting directory>
```
This script will collect (recursively) all the experiment results under `<top level plotting directory>` and generate one plot named `Result_<environment name>.png` for each environment.

### Examples
1. [2D Reacher Environment](./Experiment/Reacher2D)
2. [Fetch Peg Insertion Environment](./Experiment/FetchPegInsertion)