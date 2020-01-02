# RL + IL Research Project
Meeting Summary\
Oct 1th, 2019


## Tasks & Planning
- Use image
    - Enrionments setup.
- Use potential based
    - may be useful for some environments? Such as the car environment where you will certainly visit a different state if you do not choose the correct action.
- Use different input set for the potential function and the RL policy
    - how is this better than learning a mapping from input to potential to input to RL?
- Normalized Actor Critic and SQIL solves the same problem -> assign low reward to s,a not visited by the expert.
    - The problem is that they are not using advice so not able to recover the optimal trajectory
- Have you considered or tried using just the imitation part of the reward function as a reward, without using any real reward signal at all?
    - Saw a paper out of Levine’s lab that was doing this (SQIL) (differently)
    - Wei-di already tried this out

## Issues
- How to evaluate the performance of the potential function?
    - GAIL also learns a discriminator, maybe consider train a gail first and then train the rl+Shaping?
- Apply entropy regularized RL to shaping?

## Meeting Notes

- Hierarchical learning of RL and Imitation
    - Difference between RL and IL hierarchy
- Video demonstrations
    - or just use RGBD data
- (Image of robot, action of robot)   -   (image from human not aligned)
    - pi(z) = a z from encoder of the robot (from this point on, everything stays the same)
- The triangle where you have data from simulation, data from robot and data from human demonstrator, how do you merge, find the alignment of these three?

- Align observation data of robot with observation data from video
- GAN reconstructs robot image, we learn representation of dataset
    - Another function takes these representations and learns actions
    - Shared latent representation between robot demo images and human demo images
    - Would there be any way to inform the next data that we need to get from the simulator based on the current mapping between the human dataset and the robot dataset?

- Side note: shared autonomy
    - small input space leads to large output space

- Some papers to read
    - Sim2Real View Invariant Visual Servoing by Recurrent Control
    - One-Shot Imitation from Watching Videos
    - CycleGAN for sim2real Domain Adaptation
        - Mentioned about using human demonstrations
    - VR-Goggles for robots real-to-sim domain adaptation for visual control
        - Learn based adaptation
        - One shot sim2real transfer by running the policy on these transferred images
        - Artistic style transfer for videos - works on video sequences instead of individual frames, targeting generating temporally consistent sylizations for sequential inputs.
        - **artistic style transfer for videos and spherical images** by ruder solves shift invariant problem by adding **temporal constraints**, which is expensive. compare to **Real time neural style transfers** by huang, which gives a cheaper solution -> method wrong, actually due to the inexplicitly imposed consistency constraints for regional shifts by optical flow. -> shift loss
    - Check the deep mimic paper
    - Image transformation papers
        - pix to pix
        - cycle gan
        - CyCADA
            - with semantic constraint TODO
- Existing papers on sim 2 real, how about real demonstration to simulator? it is the same problem, but maybe real world demonstration?

## Reminder

## Experiments
