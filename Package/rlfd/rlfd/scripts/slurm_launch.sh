#!/bin/bash

#SBATCH --account=def-florian7
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10        # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --mem-per-cpu=4GB    
#SBATCH --time=01-00:00:00        # time format: day-hour:min:sec
#SBATCH --job-name=train             
#SBATCH --output=job-%x-%j.out

#SBATCH --mail-user=cheney.wu@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# Parameters (make sure it is consistent with the resources required above)
NUM_NODES=1
NUM_CPU_PER_NODE=10
NUM_GPU_PER_NODE=1
TRAINING_FILE="<parameters>.py"

# Setup
module load nixpkgs/16.09 intel/2018.3 openmpi/3.1.4 mpi4py/3.0.0 cuda/10.1 cudnn/7.6.5
source /home/yuchenwu/.bashrc
source /home/yuchenwu/rlfd/venv/bin/activate
source /home/yuchenwu/rlfd/setup.sh
# for pytorch
export LD_LIBRARY_PATH=/home/yuchenwu/rlfd/venv/lib/python3.6/site-packages/torch/lib:$LD_LIBRARY_PATH

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)  # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address | xargs)  # Making address, xargs to remove trailing spaces
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password & # Starting the head
sleep 5
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

for ((  i=1; i<$NUM_NODES; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  # Flag --block will keep ray process alive on each compute node.
  sleep 5
done

# Launch the training process
let NUM_CPUS=${NUM_CPU_PER_NODE}*${NUM_NODES} NUM_GPUS=${NUM_GPU_PER_NODE}*${NUM_NODES}
python -u -m rlfd.launch --redis_password $redis_password --ip_head $ip_head --num_cpus $NUM_CPUS --num_gpus $NUM_GPUS --targets train:${TRAINING_FILE}
