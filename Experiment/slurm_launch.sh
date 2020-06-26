#!/bin/bash

#SBATCH --account=def-florian7
#SBATCH --nodes=1
#SBATCH --ntasks=20               # number of MPI processes
#SBATCH --cpus-per-task=1         # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --gres=gpu:1              
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=01-00:00:00        # time format: day-hour:min:sec
#SBATCH --job-name=train             
#SBATCH --output=job-%x-%j.out

#SBATCH --mail-user=cheney.wu@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# Setup
module load cuda/10.1 cudnn openmpi/3.1.2 mpi4py/3.0.0
source /home/yuchenwu/.bashrc
source /home/yuchenwu/TD3fD-through-Shaping-using-Generative-Models/venv2/bin/activate
source /home/yuchenwu/TD3fD-through-Shaping-using-Generative-Models/setup.sh
# for pytorch
export LD_LIBRARY_PATH=/home/yuchenwu/TD3fD-through-Shaping-using-Generative-Models/venv2/lib/python3.6/site-packages/torch/lib:$LD_LIBRARY_PATH


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