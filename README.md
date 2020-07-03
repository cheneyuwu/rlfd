# Shaping Rewards for Combined Reinforcement and Imitation Learning using Generative Models

- by Yuchen Wu, Melissa Mozifian and Prof. Florian Shkurti

## [Link to Project Webpage](http://www.cs.toronto.edu/~florian/rl_with_shaping/)

## Installation

- setup Mujoco
  - [Instructions on installing it locally](http://www.mujoco.org/)
  - [Instructions on installing it on CC](https://docs.computecanada.ca/wiki/MuJoCo)
- download the repo
  - `git clone --recurse-submodules git@github.com:cheneyuwu/rlfd-through-Shaping-using-Generative-Models`
- build virtual env
  - cluster: `module load python/3.6` (so that you have the correct python version)
  - run `virtualenv venv` inside the root folder of the repo
- enter virtual env and install packages:
  - install mujoco_py
    - `pip install mujoco_py`
  - install tensorflow
    - local: `pip install tensorflow tensorflow_probability`
    - cluster: `pip install tensorflow_gpu tensorflow_probability`
  - install pytorch (not used, but in case)
    - `pip install torch torchvision torchsummary`
  - install ray
    - `pip install ray`
    - `pip install ray[tune]`
  - install environments
    - `pip install gym`
    - `pip install -e Package/gym_rlfd`
    - `pip install -e Package/d4rl`
    - `pip install -e Package/metaworld --no-deps`
  - install rlfd
    - `pip install -e Package/rlfd`

## Running Experiments

### Generating Demonstrations (Not Available Right Now)

Demonstration data must be stored as a `npz` file named `demo_data.npz`, it should contain entries: `(o, ag, g, u, o_2, ag_2, g_2, r, done)`, each entry is an array of shape `(number of episodes, episode_length, dimension of entry)`

For our experiments, we provide two options for generating demonstration data.

1. Using hard coded scripts. Refer to `<root>/Experiment/<envrionment names>/README.md` for more information (TODO: Document this).
2. Using pre-trained policies. Our implementation of TD3 stores models through pickle as `<policy name>.pkl`. To use a pre-trained model to generate demonstration data:

- create a json file named `demo_config.json` containing the following:

  ```json
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

- put the pre-trained `<policy name>.pkl` in the same folder as your `demo_config.json` and then run

  ```bash
  python -m rlfd.launch --targets demo --policy_file <policy name>.pkl
  ```

### Training Models

- Create a python script storing parameters to be used for training your model, named `<param name>.py`
  - We have pre-defined parameter files for the environments we tested in `<root>/Package/rlfd/rlfd/td3/params/*` and
  - Optionally, in the python file you just created, you can simple import from our pre-defined parameters, e.g.

    ```python
    from copy import deepcopy
    from rlfd.td3.params.default_params import parameters

    params_config = deepcopy(parameters)
    params_config["ddpg"]["demo_strategy"] = ("gan", "none")
    params_config["seed"] = tuple(range(10))
    ```

    Note 1: the name of the global variable starts with `params_config`. \
    Note 2: if a parameter is of type `tuple`, it is assumed that you want to iterate over the tuple and train multiple agents each with a different parameter listed in the `tuple`. \
    In the above script, we imported the default parameters used for the 2D Peg Insertion environment, then we override some parameters:

    - `params_config["ddpg"]["demo_strategy"] = ("gan", "none")` means to train two agents, one with reward shaping via GAN, one with just TD3.
    - `params_config["seed"] = tuple(range(10))` means to train 10 agents with seed 0 to 10.

    Therefore, this modified parameter dictionary tries to run 20 experiments, TD3 + GAN shaping with seed 0-10 and TD3 with seed 0-10. \
    See `<root>/Package/rlfd/rlfd/td3/config` for a detailed description of all parameters.
- Put the `demo_data.npz` generated in previous step in the same folder as `<param name>.py`
  - Note: If you only want to run our implementation of TD3, then `demo_data.npz` is not needed.
- In the same folder, run

  ```bash
  python -m rlfd.launch --targets train:<param name>.py --num_cpus <default to 1> --num_gpus <default to 0>
  ```

- After training, the following files should present in the each experiment directory:

  ```txt
  demo_data.npz - demontration data, which is copies to each run
  log.txt       - training logs
  params.json   - parameters for the experiment enclosed in this directory
  progress.csv  - experiment statistics for plotting
  policies      - a folder containing intermediate policies, <policy_[0-9]+>.pkl, and the last one, policy_last.pkl
  ```

### Evaluating/Visualizing Models

```bash
python -m rlfd.launch --targets evaluate --policy <policy file name>.pkl
```

### Plotting

1. Use tensorboard

  ```bash
  tensorboard --logdir <path to experiment directory> --port <port number>
  ```

  Then open `localhost:<port number>`
2. Use our own plotting scripts

  ```bash
  python -m rlfd.launch --targets plot --exp_dir <top level plotting directory>
  ```

This script will collect (recursively) all the experiment results under `<top level plotting directory>` and generate one plot named `Result_<environment name>.png` for each environment.

### Examples

1. [2D Peg Insertion Task with Optimal Demonstrations](./Experiment/YWPegInHole2DOptDemo)
