# HowTos

## Installation
1. setup Mujoco
  - local: http://www.mujoco.org/
  - cluster: https://docs.computecanada.ca/wiki/MuJoCo
2. download the repo
  - `git clone --recurse-submodules git@github.com:cheneyuwu/TD3fD-through-Shaping-using-Generative-Models`
3. build virtual env
  - cluster: `module load python/3.6` (make sure that you have the correct python version)
  - run `virtualenv venv` inside the root folder of the repo
4. enter virtual env and install packages:
  - install mujoco_py, mpi4py
    - `pip install mujoco_py==1.50.1.68`
    - local: `pip install mpi4py`
    - cluster: `module load mpi4py`
  - install tensorflow
    - local: `pip install tensorflow tensorflow_probability`
    - cluster: `pip isntall tensorflow_gpu tensorflow_probability`
  - install pytorch (not used, but in case)
    - `pip install torch==1.3.1 torchvision torchtext torchaudio torchsummary`
      - Note: i think pytorch 1.4 has some issue running on cluster, so don't use it for now.
  - (goto Package folder) install gym and metaworld (and maybe other environments)
    - `pip install -e gym --no-deps`
    - `pip install -e metaworld --no-deps`
    - Optional: `pip install dm-control dm-env --no-deps` (for DeepMind Control Suite)
      - if you install dm_control, also install the dmc to gym wrapper (dmc2gym) included in the submodule of this repo
  - (goto Package folder) install rlfd
    - `pip install -e rlfd --no-deps`
  - extra packages that should be installed
    - `pip install certifi chardet urllib3 requests idna gtimer matplotlib pandas future lxml pyopengl`