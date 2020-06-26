# Peg Insertion with Optimal Demonstrations

Demonstrations have been generated and stored in this folder, `demo_data.npz`. It was generated using the script `rlfd/rlfd/demo_utils/generate_demo_fetch_peg_in_hole_2d.py`.

There are several parameter files, e.g. td3_bc.py, that run different experiments, take a look.

Commands below should be run in this directory.

- Run an experiment

  ```bash
    mpirun -n <number of threads> python -m rlfd.launch --targets train:<parameter file>.py
  ```

  Number of threads depends on number of experiments running in parallel.

- Check plots through tensorboard

  ```bash
    tensorboard --logdir . --port 2222
  ```

- Generate plots with mean and variance using our plotting script

  ```bash
    python -m rlfd.launch --targets plot
  ```

  It is possible to generate plots with arbitrary x/y axis. Check the `launch.py` script for more info.

- Visualize learned policy

  ```bash
  python -m rlfd.launch --targets evaluate --policy <path to stored policy, file name should be something like "policy_latest.pkl">
  ```
