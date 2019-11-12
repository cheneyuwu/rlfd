# Pick and Place Environment

## Running Experiments

### Generating Demonstrations
The demonstration data for this environment has been stored in `demo_data.npz`. It is generated through a hard coded script `generate_demo_data.py`. However, during testing, we randomnized initial pose of the robot arm but fixed the target position of the object. With this change, the "goal" of this environment is fixed and we thereby removed the notion of "goal" for this environment.

### Training Models
- The default parameters for this environment is stored in `<root>/Package/td3fd/td3fd/ddpg/param/ddpg_pick_and_place_rand_init.py`.
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
Note: this will take a **long** time (hours) to finish since GAN and MAF training are slow. You may also consider running each experiment inidividually.

### Evaluating/Visualizing Models
You can visualize each model using the following command
```
python -m td3fd.launch --targets evaluate --policy_file config_<config name>/policies/policy_latest.pkl
```

### Plotting
```
python -m td3fd.launch --targets plot
```
A plot named `Result_YWFetchPickAndPlaceRandInit-v0.png` will be generated.
The left plot shows success rate vs epochs, and the right plot shows total reward vs epochs. In this case, as we use sparse reward where the agent gets a reward of -1 unless it reaches the goal, these two plots look the same.