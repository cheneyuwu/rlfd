# HowTos

## Install this package
1. setup Mujoco
  - On cluster: https://docs.computecanada.ca/wiki/MuJoCo
  - note: you need to manually install mujoco_py
2. download the repo (modify .gitmodules if necessary!)
  - `git clone git@github.com:cheneyuwu/RLProject`
  - `git submodule init`
  - `git submodule sync`
  - `git submodule update --remote`
3. enter virtual env and install packages: gym, yw, tensorflow, tensorflow_probability
  - note: on compute canada cluster you need to manually install tensorflow and use `module load mpi4py` to get the mpi python package
4. add rl project directory to the bashrc
  - export RLProject='/home/yuchenwu/projects/def-florian7/yuchenwu/RLProject'
5. Modify the directory of EXPRUN according to the true exp running directory

## Compute Canada Cluster
- [Wiki](https://docs.computecanada.ca/wiki/Main_Page)
- Connect
  - `ssh -Y yuchenwu@[cedar, beluga, graham].computecanada.ca`
  - Password: @0413Wuyuchen
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
  4. build virtual env
    - `module load python/3.6` (make sure that you have the correct python version)
    - create a folder `~/ENV` and run `virtualenv <env name>` inside this folder
  5. Install this package (see above)


- [SLURM](https://www.rc.fas.harvard.edu/resources/documentation/convenient-slurm-commands/)
  - Template
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
  - Run jobs
    - `sbatch *.sh`

  - Modules
    - `module list`