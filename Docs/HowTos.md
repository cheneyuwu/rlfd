# Compute Canada Cluster
- [Wiki](https://docs.computecanada.ca/wiki/Main_Page)
- Connect
  - `ssh -Y yuchenwu@cedar.computecanada.ca`
  - Password: @0413Wuyuchen
- Build environment
  - module load python/3.6
  - Use virtual env
  - pip install ...
    - numpy --no-index
    - tensorflow_gpu

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