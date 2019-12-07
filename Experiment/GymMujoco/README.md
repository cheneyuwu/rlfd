# Reacher 2D Environment

## Running Experiments

### Training Models with RL
- The default parameters for this environment is stored in `<root>/Package/td3fd/td3fd/td3/params/td3.py`.
- In this directory, we have created several parameter files that import the default parameters, modify it to run with different configurations, e.g. `train_rl.py`
Use the following command to train all models:
```
python -m td3fd.launch --targets train:train_rl.py
```
- You can stop the training `ctrl+c` once you see it converges (reward_per_eps=1e+3)

### Evaluating/Visualizing Models
You can visualize each model using the following command
```
python -m td3fd.launch --targets evaluate --policy <exp dir>/policies/policy_latest.pkl
```

### Generating Demonstrations
For this environment, we use the pre-trained policy `policy_latest.pkl` to generate demonstration data. In `demo_config.json`, we specify the following:
```
{
    "seed": 0,
    "num_eps": 40,              # Generate 40 episodes
    "fix_T": false,             # Episode length not fixed. This should be used for MuJoCo envs
    "max_concurrency": 10,      # 10 environments running in parallel
    "demo": {
        "random_eps": 0.0,      # No random actions
        "noise_eps": 0.1,       # Add Gaussian noise with var=0.1 to actions
        "render": false
    },
    "filename": "demo_data.npz"
}
```
Use the following command to generate demonstration data:
```
python -m td3fd.launch --targets demo --policy <exp directory>/policies/policy_latest.pkl
```
This command will generate `demo_data.npz` storing the generated demonstrations.

### Training Models with RL+Demo
- Now follow the comments in `train_rl.py` to use `bc` (behavior cloning) and `gan` shaping. Compare the result
Use the following command to train all models:
```
python -m td3fd.launch --targets train:train_rl.py
```

### Plotting
```
python -m td3fd.launch --targets plot
```