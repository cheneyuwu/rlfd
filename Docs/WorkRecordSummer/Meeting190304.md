# Undergrad Research Project
Meeting Summary<br>
March 4th, 2019

## TODO before meeting
- Find a way to get $Q_D$ for Mujoco environments, this may help for generating useful demonstration data.
    - Its time horizon can either be finite or infinite.
    - You can do this by train a RL agent first and get demo data from that pre-trained RL agent.
- Supervised learning for demonstration. Use Edward for BNN or Ensemble.
- Code for comparing distributions.
  - This is essentially just comparing two normal distributions.

## Command to run
- Generate demo data of 100 points and 10 points on a arbitrary function
  - python generate_demo.py none
  - note change num_itr to 100 and 10 respectively
- the demo_data_size and demo_batch_size are better to be the same and they should be the same as demo data size (10)
  

## Reminder
- UTEA Application Form