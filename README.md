# IL + RL Project
- 2019 Summer Research with Professor Florian Shkurti


## Installation
- clone the repository
- In the root directory
    - `source setup.sh`
- Go to Package/yw
    - `pip install -e .`
- Run flow tests using command
    - `ywregtest`


## Experiments
- Experiment/train.py - Main script for training the RL agent with/without demonstration.


## Algorithm Options and Flags (train_ddpg_main.py)

### Path configuration
- Both are implemented by OpenAI. Using OpenAI logger and MPI.
- Parameters
  - |||
    |-|-|
    | logdir | Log directory|
    | loglevel | Use 1 for debugging and 2 for normal information output

### Environment
- Select the environment based on user inputs. The environment should have properties listed in the env_manager.py.
- Parameters
    - |||
      |-|-|
      | env_name/env | Choose the environment. |
      | r_scale | This will scale down the reward by r_scale. The purpose is to reduce the size of the output so that demonstration can be effective early. |
      | r_shift | This will shift up the reward y r_shift. Same idea as above. |

### DDPG
staging_tf    staged values
buffer_ph_tf  placeholders

demo_q_mean_tf
demo_q_var_tf
demo_q_sample_tf
max_q_tf

Q_loss_tf -> a list of losses for each critic sample
pi_loss_tf -> a list of losses for each pi sample

policy.
pi_tf  list of output from actor
Q_tf
Q_pi_tf
Q_pi_mean_tf
Q_pi_var_tf
_input_Q

