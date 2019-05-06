# RL Project
- 2019 Summer Research with Professor Florian Shkurti

## Experiments
- train_ddpg_main.py - Main script for training the RL agent with/without demonstration.

## Algorithm Options and Flags (train_ddpg_main.py)
### Path configuration and Number of CPU
- Both are implemented by OpenAI. Using OpenAI logger and MPI.
- Parameters
  - |||
    |-|-|
    | logdir | Log directory|
    | loglevel | Use 1 for debugging and 2 for normal information output
    | num_cpu | Number of CPU cores to run this script. It just generates more rollouts and averages the weight update
### Environment
- Select the environment based on user inputs. The environment should have properties listed in the env_manager.py.
- Parameters
    - |||
      |-|-|
      | env_name/env | Choose the environment. |
      | r_scale | This will scale down the reward by r_scale. The purpose is to reduce the size of the output so that demonstration can be effective early. |
      | r_shift | This will shift up the reward y r_shift. Same idea as above. |
### Demonstration Neural Network
- Configuration Parameters
    - |||
      |-|-|
      | demo_config | Name of the demonstration neural network. |
      | demo_scope | Variable scope of the demonstration neural network. |
      | demo_net_type | Choose between "ensemble" and "baysian" |
      | demo_sample | Under "ensemble", this will be the number of neural network instances. Under "baysian", this will be the number of samples drawn from the   baysian neural network during testing.
      | demo_layers | Number of hidden layers of the neural network. |
      | demo_hidden | Number of hidden units each layer has in the neurla network. |
      | demo_lr | Learning rate of the demonstration neural network. |
    - Note: Parameters start with "demo" will be used only for demonstration. Others will be shared by other modules.
- Training Parameters
    - |||
      |-|-|
      | dm_epochs | Number of epochs |
      | demo_data_size | Training data size in number of transitions. |
      | demo_test_batch_size | Testing data size in number of transitions. |
      | T | Length of each episode. Need this for storing data into the replay buffer. |
      | max_u | Maximum magnitude of action |
      | buffer_size | Size of the replay buffer. Not quit related for demonstration training. Just use some large number. |
      | input_dims | Input dimensions of the environment. |
      | sample_transitions | A simple function to sample transitions from the replay buffer. Not quit related to the demonstration training. Just because I use   replay_buffer to store the training data. |
    - Note: Parameters start with "demo" will be used only for demonstration. Others will be shared by other modules.



