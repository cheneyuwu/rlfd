#!/bin/bash

## THIS IS A TEMPLATE SLURM LAUNCH SCRIPT TO BE MODIFIED

#SBATCH --account=def-florian7
#SBATCH --nodes=%%NODES%%
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=%%CPUS_PER_NODE%%  # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --gres=gpu:%%GPUS_PER_NODE%%       # request GPU "generic resource"
#SBATCH --mem-per-cpu=%%MEM_PER_CPU%%GB
#SBATCH --time=00-%%TIME%%:00:00           # time format: day-hour:min:sec
#SBATCH --job-name=%%NAME%%
#SBATCH --output=job-%x-%j.out

#SBATCH --mail-user=cheney.wu@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# Parameters (make sure it is consistent with the resources required above)
NUM_NODES=%%NODES%%
NUM_CPU_PER_NODE=%%CPUS_PER_NODE%%
NUM_GPU_PER_NODE=%%GPUS_PER_NODE%%
TRAINING_FILE=%%CONFIG_FILE%%

# Setup
module load nixpkgs/16.09 intel/2018.3 openmpi/3.1.4 mpi4py/3.0.0 cuda/10.1 cudnn/7.6.5
source /home/yuchenwu/.bashrc
source /home/yuchenwu/rlfd/venv/bin/activate
source /home/yuchenwu/rlfd/setup.sh
# for pytorch
export LD_LIBRARY_PATH=/home/yuchenwu/rlfd/venv/lib/python3.6/site-packages/torch/lib:$LD_LIBRARY_PATH

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)  # Getting the node names
nodes_array=( $nodes )

# Find the first available port
port=%%PORT%%
port_available=true
while true
do
  for (( i=0; i<$NUM_NODES; i++ ))
  do
    node=${nodes_array[$i]}
    if ( srun --nodes=1 --ntasks=1 -w $node ss -lntup | grep :${port}  > /dev/null )
    then
      echo "${port} not available."
      port_available=false
      break
    fi
  done
  if [ ${port_available} = true ]
  then
    echo "Use port: $port"
    break
  else
    ((port=port+1))
    port_available=true
  fi
done

# Get the ip address for the head node (the node where we run this script) and pass it to other nodes
node1=${nodes_array[0]}
# ip_address=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address | xargs)  # Making address, xargs to remove trailing spaces
ip_address=$(python -c "import ray.services; print(ray.services.get_node_ip_address())")

ip_address_port=${ip_address}:${port}
redis_password=$(uuidgen)

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=$port --redis-password=$redis_password --num-cpus ${NUM_CPU_PER_NODE} --num-gpus ${NUM_GPU_PER_NODE} & # Starting the head
sleep 5
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

for (( i=1; i<$NUM_NODES; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_address_port --redis-password=$redis_password --num-cpus ${NUM_CPU_PER_NODE} --num-gpus ${NUM_GPU_PER_NODE} & # Starting the workers
  # Flag --block will keep ray process alive on each compute node.
  sleep 5
done

# Launch the training process
let NUM_CPUS=${NUM_CPU_PER_NODE}*${NUM_NODES} NUM_GPUS=${NUM_GPU_PER_NODE}*${NUM_NODES}
python -u -m rlfd.launch --redis_password $redis_password --ip_head $ip_address_port --num_cpus $NUM_CPUS --num_gpus $NUM_GPUS --targets train:${TRAINING_FILE}
