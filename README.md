# [rlfd]((http://www.cs.toronto.edu/~florian/rl_with_shaping/))

## Installation

- setup Mujoco
  - [Instructions on installing it locally](http://www.mujoco.org/)
  - [Instructions on installing it on CC](https://docs.computecanada.ca/wiki/MuJoCo)
- download the repo
  - `git clone --recurse-submodules git@github.com:cheneyuwu/rlfd`
- build virtual env (python >= 3.6, < 3.8)
  - cluster: `module load python/3.6` (so you have the correct version)
  - `virtualenv venv` (or use conda if preferred)
- enter virtual env and install packages:
  - install mujoco_py (>= 2.0.0)
    - `pip install mujoco_py`
  - install tensorflow (>= 2.2.0)
    - local: `pip install tensorflow tensorflow_probability` (or use conda if preferred)
    - cluster: `pip install tensorflow_gpu tensorflow_probability`
  - install pytorch (>= 1.5.0)
    - `pip install torch torchvision` (or use conda if preferred)
  - install ray with tune (>= 0.8.0)
    - `pip install ray[tune]`
  - install environments and rlfd
    - `pip install gym`
    - `pip install -e gym_rlfd`
    - `pip install -e d4rl`
  - install rlfd
    - `pip install -e rlfd`

## Running Experiments

### Train an agent

Example launch files will be provided when we settle down our methods. For now, **carefully go through `rlfd/launch.py` and see how a launch file is parsed.**

A launch file should look like this:

```python
# file name: <launch file>.py
from copy import deepcopy
from rlfd.params.sac import gym_mujoco_params  # import default parameters for gym_mujoco environments
# the launch file defines a global dict called params_config
params_config = deepcopy(gym_mujoco_params)  # get default parameters of an algorithm.
params_config["config"] = ("SAC", )  # provide your exp config name, will be used for plotting.
params_config["env_name"] = "halfcheetah-medium-v0"  # which environment
# Make whatever changes to the default parameters.
params_config["seed"] = tuple(range(5))  # a tuple means grid search.
```

Train locally:

```bash
python -m rlfd.launch --targets train:<param name>.py --num_cpus <default to 1> --num_gpus <default to 0>
```

Train on CC:

```bash
python -m rlfd.launch --targets slurm:<param name>.py --num_cpus <default to 1> --num_gpus <default to 0> --memory <per cpu, default to 4GB>
```

After training, the following files should present in the each experiment directory:

```txt
params.json   - parameters for the experiment enclosed in this directory
policies      - a folder containing intermediate policies and the last one after online/offline training.
summaries     - tensorboard summaries
```

### Evaluate / Visualze

```bash
python -m rlfd.launch --targets evaluate --policy <policy file name>.pkl
```

### Plotting

Use tensorboard

```bash
tensorboard --logdir <path to experiment directory> --port <port number>
```

Use our own plotting scripts, check `rlfd/plot.py` for details.

```bash
python -m rlfd.launch --targets plot --exp_dir <top level plotting directory>
```
