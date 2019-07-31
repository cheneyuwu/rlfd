#!/bin/bash

#SBATCH --nodes=2
#SBATCH --gres=gpu:2                      # request GPU "generic resource"
#SBATCH --cpus-per-task=1                 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --ntasks=20                       # number of MPI processes
#SBATCH --mem-per-cpu=4G                  # memory per node
#SBATCH --time=00-10:00:00                # time format: day-hour:min:sec
#SBATCH --output=fetchreach-%N-%j.out     # %N for node name, %j for jobID
#SBATCH --account=def-florian7

#SBATCH --mail-user=cheney.wu@mail.utoronto.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

# Setup
module load nixpkgs/16.09  intel/2018.3  cuda/10.0  cudnn/7.5  python/3.6  openmpi/3.1.2  mpi4py/3.0.0
source /home/yuchenwu/.bashrc
source /home/yuchenwu/ENV/tf114/bin/activate
nvidia-smi --compute-mode=0
source /home/yuchenwu/projects/def-florian7/yuchenwu/RLProject/setup.sh

# LAUNCH_EXP_DIR="/home/yuchenwu/scratch/OpenAI/Temp"
LAUNCH_EXP_DIR="."
N_TASKS=20

mpiexec -n ${N_TASKS} python -m yw.flow.launch --exp_dir ${LAUNCH_EXP_DIR} --targets train:rl.py # change this name
# train:rldense.py demo
# train:rlmaf.py
# train:rlrbmaf.py
# train:rlmerged.py