#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1                      # request GPU "generic resource"
#SBATCH --cpus-per-task=1                 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --ntasks=5                       # number of MPI processes
#SBATCH --mem-per-cpu=6G                  # memory per node
#SBATCH --time=00-08:00:00                # time format: day-hour:min:sec
#SBATCH --output=bc-%N-%j.out        # %N for node name, %j for jobID

#SBATCH --account=def-florian7

# Setup
module load nixpkgs/16.09  intel/2018.3  cuda/10.0  cudnn/7.5  python/3.6  openmpi/3.1.2  mpi4py/3.0.0
source ~/.bashrc
source ~/ENV/tf114/bin/activate
nvidia-smi --compute-mode=0
source ${RLProject}/setup.sh

N_TASKS=5
# LAUNCH_EXP_DIR="/home/yuchenwu/scratch/OpenAI/Temp"
LAUNCH_EXP_DIR="."
TRAINING_FILE="train_pure_bc.py"

mpiexec -n ${N_TASKS} python -m td3fd.launch --exp_dir ${LAUNCH_EXP_DIR} --targets train:${TRAINING_FILE}