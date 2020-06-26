# HowTos

## Install this package
1. setup Mujoco
  - On cluster: https://docs.computecanada.ca/wiki/MuJoCo
  - note: you need to manually install **mujoco_py**
2. download the repo
  - `git clone git@github.com:cheneyuwu/TD3fD-through-Shaping-using-Generative-Models`
    - (modify .gitmodules if necessary!)
  - `git submodule init`
  - `git submodule sync`
  - `git submodule update --remote`
3. build virtual env
  - On cluster: `module load python/3.6` (make sure that you have the correct python version)
  - run `virtualenv venv` inside the root folder of the repo
4. enter virtual env and install packages:
  - install mujoco_py, mpi4py
    - `pip install mujoco_py==1.50.1.68`
    - `pip install mpi4py`
  - install tensorflow with
    - For tf1: `pip install tensorflow-gpu==1.14.1 tensorflow-probability==0.7.0 tensorflow-determinism`
    - For tf2: `pip install tensorflow tensorflow_probability`
      - Note: on cluster use `tensorflow_gpu`
  - install pytorch with
    - `pip install torch==1.3.1 torchvision torchtext torchaudio torchsummary`
  - install gym and metaworld and other environments
    - cd to gym and metaworld then `pip install -e . --no-deps`
    - for dm_control, install it directly via `pip install dm_control --no-deps`
    - also install the dmc to gym wrapper (dmc2gym) included in the submodule of this repo
  - install rlfd and rlkit
    - cd to rlfd and rlkit then `pip install -e . --no-deps`
  - extra packages that should be installed:
    - `pip install black certifi chardet urllib3 requests idna gtimer matplotlib pandas future lxml pyopengl`
    - `pip install dm-env dm-control --no-deps`
  - note: on compute canada cluster you need to manually install tensorflow and use `module load mpi4py` to get the mpi python package
5. Modify the directory of EXPRUN according to the true exp running directory
6. add rl project directory to the bashrc
  - export RLProject='/home/yuchenwu/projects/def-florian7/yuchenwu/RLProject'

## Compute Canada Cluster
- [Wiki](https://docs.computecanada.ca/wiki/Main_Page)
- Connect
  - `ssh -Y yuchenwu@[cedar, beluga, graham].computecanada.ca`
  - add this to the `~/.ssh/config` for easy connect: `ssh [cedar, beluga, graham]`
- Setup
  1. setup ssh public key authentication
    - `scp ~/.ssh/id_rsa.pub [cedar, beluga, graham]:~`
    - create file `~/.ssh/authorized_keys` on server, and then copy the content of `id_rsa.pub` to this file
    - `chmod 600 /home/USERNAME/.ssh/authorized_keys`
    - `chmod 700 /home/USERNAME/.ssh`
  2. generate ssh on the server and add it to the github account
    - https://help.github.com/en/enterprise/2.15/user/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
    - For multiple ssh keys: https://gist.github.com/jexchan/2351996
  3. clone github repository and setup bash, vim, tmux, etc
    - note that compute canada's bashrc is not empty, do not overwrite the original content
  4. Install this package (see above)


- [SLURM](https://www.rc.fas.harvard.edu/resources/documentation/convenient-slurm-commands/)
  - Sbatch Template
  ```
  #!/bin/bash
  #SBATCH --gres=gpu:1        # request GPU "generic resource"
  #SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
  #SBATCH --mem=32000M        # memory per node
  #SBATCH --time=0-03:00      # time (DD-HH:MM)
  #SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

  module load cuda cudnn
  source <directory to your virtual env>
  python ./tensorflow-test.py
  ```
  - Run jobs: `sbatch *.sh`
  - Or use salloc: `salloc --mem-per-cpu=8G --ntasks 1 --nodes 1 --cpus-per-task=1 --gres=gpu:1`

  - Modules
    - `module list`