# Reacher 2D Environment

This is a toy 2d point reaching environment where you simply direct a point to its target location.
```
state  - (x, y), current location of the point
goal   - (x', y'), goal location
action - (dx, dy), direction to move
reward - 0 if ||(x, y) - (x', y')|| < threshold else -1
```
The updating rule is simply: `state_t+1 = state_t + action_t`.

## Running Experiments

### Generating Demonstrations
For this environment, we use a pre-trained policy `demo_policy.pkl` to generate demonstration data. In `demo_config.json`, we specify the following:
```
{
    "seed": 0,
    "num_eps": 40,              # Generate 40 episodes
    "fix_T": true,              # Episode length fixed, so demonstrations are stored in form (num_episode, episode_length, dimension)
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
python -m td3fd.launch --targets demo --policy_file demo_policy.pkl
```
This command will generate `demo_data.npz` storing the generated demonstrations.

### Training Models
- The default parameters for this environment is stored in `<root>/Package/td3fd/td3fd/ddpg/param/ddpg_reacher.py`.
- In this directory, we have created several parameter files that import the default parameters, modify it to run with different configurations, e.g. `train_gan.py`
```
params_config = deepcopy(base_params)              # copy base parameters
params_config["config"] = ("TD3_GAN_Shaping",)     # change config name to "TD3_GAN_Shaping"
params_config["ddpg"]["demo_strategy"] = "gan"     # use GAN Shaping
params_config["ddpg"]["sample_demo_buffer"] = True # use demonstrations to train policy (actor)
params_config["seed"] = 0                          # seed everything with 0
```
Use the following command to train all models:
```
python -m td3fd.launch --targets train:train_rl.py train:train_rlbc.py train:train_gan.py train:train_maf.py
```
This command will train the following 4 models, each stored in an individual folder:
```
config_TD3             - TD3 without demonstrations
config_TD3_GAN_Shaping - TD3 + Reward shaping via GAN
config_TD3_MAF_Shaping - TD3 + Reward shaping via normalizing flow (MAF)
config_TD3_BC          - TD3 + Behavioral cloning
```

### Evaluating/Visualizing Models
You can visualize each model using the following command
```
python -m td3fd.launch --targets evaluate --policy_file config_<config name>/policies/policy_latest.pkl
```

### Plotting
```
python -m td3fd.launch --targets plot
```
A plot named `Result_Reach2DF.png` will be generated, which should look like ![Reference_Result_Reach2DF](./Reference_Result_Reach2DF.png)
The left plot shows success rate vs epochs, and the right plot shows total reward vs epochs. In this case, as we use sparse reward where the agent gets a reward of -1 unless it reaches the goal, these two plots look the same.