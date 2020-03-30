#!/bin/bash

# A SAMPLE SLURM SCRIPT

#SBATCH --nodes=1
#SBATCH --gres=gpu:1                      # request GPU "generic resource"
#SBATCH --cpus-per-task=1                 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --ntasks=20                       # number of MPI processes
#SBATCH --mem-per-cpu=8G                  # memory per node
#SBATCH --time=01-00:00:00                # time format: day-hour:min:sec
#SBATCH --output=exp_name-%N-%j.out       # %N for node name, %j for jobID

#SBATCH --account=def-florian7

#SBATCH --mail-user=cheney.wu@mail.utoronto.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

# Setup
module load nixpkgs/16.09  intel/2018.3  cuda/10.0  cudnn/7.5  python/3.6  openmpi/3.1.2  mpi4py/3.0.0
source /home/yuchenwu/.bashrc
source /home/yuchenwu/TD3fD-through-Shaping-using-Generative-Models/venv/bin/activate
nvidia-smi --compute-mode=0
source ${RLProject}/setup.sh
# for pytorch
export LD_LIBRARY_PATH=/home/yuchenwu/TD3fD-through-Shaping-using-Generative-Models/venv/lib/python3.6/site-packages/torch/lib:$LD_LIBRARY_PATH


N_TASKS=20
LAUNCH_EXP_DIR="."
TRAINING_FILE="<parameters>.py"

mpiexec -n ${N_TASKS} python -m td3fd.launch --exp_dir ${LAUNCH_EXP_DIR} --targets train:${TRAINING_FILE}


# SOME USEFUL COMMANDS

# 1. change config name for seeds that succeeded
# find . -name log.txt | xargs grep -il -m 1 "new best success rate: 1.0" | sed -e "s/log.txt/params_renamed.json/" | xargs sed -i 's/", "env_name/-succeed", "env_name/'

# 2. recursively find a string on certain file then change:
# find . -name '*.py' | xargs grep 'source'
# find . -name '*.py' | xargs sed -i 's/source/target/'