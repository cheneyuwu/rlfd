## List of experiments to run in this directory
python -m rlfd.launch --targets slurm:cql_sac.py --num_cpus  4 --num_gpus 1 --memory 4 --time 24
python -m rlfd.launch --targets slurm:sac.py     --num_cpus  4 --num_gpus 1 --memory 4 --time 24
python -m rlfd.launch --targets slurm:sac_nf.py  --num_cpus  4 --num_gpus 1 --memory 8 --time 64
python -m rlfd.launch --targets slurm:sac_gan.py --num_cpus  4 --num_gpus 1 --memory 6 --time 36
python -m rlfd.launch --targets slurm:td3.py     --num_cpus  4 --num_gpus 1 --memory 4 --time 24
python -m rlfd.launch --targets slurm:td3_nf.py  --num_cpus  4 --num_gpus 1 --memory 8 --time 64
python -m rlfd.launch --targets slurm:td3_gan.py --num_cpus  4 --num_gpus 1 --memory 6 --time 36