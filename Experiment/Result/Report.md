# Experiment Variables to Consider
## Rl Training
- How many RL samples should we use?

## Select demonstration
1. Directly use a well trained critic function
2. Train a demonstration neural network
    - How many Demo samples should we use?
    - Should we sample with replacement?

## How to form the training set for demonstration
- How many transitions should we use in the training set. training set size.
- What transitions to select?
    - Select the entire episode of an expert demonstration
        - Random select some data and its nearby points. -> The action is still selected by the expert.
    - Randomly select some data points. -> The action is selected by the expert.

## Combine RL and Demonstration: Basic formula is $\lambda Q_{RL} + (1-\lambda) Q_{Demo}$
- Calibrate $\lambda$ using time
    - $\lambda = 1$ as time -> infinity and $\lambda \approx 0$ as time -> 0
- Calibrate $\lambda$ using demonstration uncertainty only
    1. $\lambda=e^{- a * v^2}$ where $a$ is a hyperparameter decided based on the average uncertainty from demonstration nn.


# Result && Report

## Reach2D Env
- **May03R2D**
    - No Modification to the environment
    - Use only 1 RL sample as we do not care about the uncertainty from RL critic
    - Always sample with replacement
    - If applicable, use 12 demonstration samples. Later we can try more of this.
    - Only use uncertainty from demonstration nn only, see no.1 from above.
    - **CriticAsDemo**
        - Use 12 RL samples to train the pre-RL first. Check the uncertainty from the demonstration neural network.
        - Always use the max value between the bellman target and the weighted combination
        - The finally trained RL should have only 1 sample.
        - **TimeInvar**
            - Converged well. Since the estimated Q value is not the real one (it is far below the actual Q value), time var and time invar should not have much difference.
        - **TimeVariant**
            - See above.
    - **ExpertAsDemo**
        - **TimeInvar**
            - **Size1024**
                - **Random**
                    - **Sample12**
                        - Never converged. success rate is at 0.2
                    - **Sample36**
                - **Nearby**
                    - **Sample12**
                        - Never converged. Simiar to the Ramdom version.
                    - **Sample36**
            - **Size4096**
                - **Random**
                    - **Sample12**
                        - Initially performed well. later becomes 0.1.
                    - **Sample36**
                - **Nearby**
                    - **Sample12**
                        - Initially performed very well! later becomes 0.3
                    - **Sample36**
        - **TimeVariant**

    - **CriticAsDemo**
    -

## Reach2DFirstOrder
- **M31R2DFO**
    - Experiment using the Reach2DFirstOrder environment. The goal is fixed at (0,0)
    - **NegativeReward**: Suppose that the distance between current position and the goal position (0,0) is $d$, the reward is $(-d + 0.5)/12$. I added a small shift (0.5) to the reward so that demonstration neural net can affect training of ddpg critic at the very beginning.
        - **CriticAsDemo**: Use a trained DDPG critic as the demonstrator.
            - **DirectUpdate**: $q(s_t, a) = compare(r + q_r(s_{t+1},a'), q_d(s_t,a))$.
                - **NoInterfere**: The demonstraion nn is trivially added. RL critic won't learn anything from the demonstration. Just want to see whether the rl_q and the demo_q becomes similar at the end of the training.
                    - *Result&Comment*: The success rate over epochs went to 100% quickly. The final $q_r$ looks similar to $q_d$. This result indicates that the ddpg critic converges when not using demonstration.
                - **Count**: Count how many outputs $q_d^k$ are greater than $q_r$
                    - *Result&Comment*: The success rate over epochs went to 100% quickly. $q_r$ did learn from $q_d$, as shown in the query.mp4. $q_d$ did not break the gradient of $q_r$ and it helped the actor learn the correct action. This result shows that our method works!
        - **DemoNNAsDemo**: The demonstration neural net is trained through supervised learning given lots of expert transition tuples $(s_t, a, s_{t+1}, q)$ where $q(s_t,a_t) = r_t + q(s_{t+1}, a_{t+1})$. The $q$ is manually calculated for each transition in each episode.
            - **DirectUpdate**: $q(s_t, a) = compare(r + q_r(s_{t+1},a'), q_d(s_t,a))$.
                - **NoInterfere**: The demonstraion nn is trivially added. RL critic won't learn anything from the demonstration. Just want to see whether the rl_q and the demo_q becomes the same at the end of the training.
                    - *Result&Comment*: The success rate over epochs went to 100% quickly. However, although the shape of $q_r$ looks similar to $q_d$ at the end (like a "dome", which is expected.), It is smaller than $q_d$ on average. This result indicates that, the actor can learn the correct action for each input state as long as the q estimate from critic has the correct shape(trend).
                - **Count**: Count how many outputs $q_d^k$ are greater than $q_r$
                    - *Result&Comment*: The success rate over epochs is quite noisy. When checking the V(s,a) in query.mp4, we can see that the V(s,a) is changes a lot over epochs, and it does not have the desired shape ("dome"). It is very likely because the train demonstration neural network provides incorrect estimate when a particular combination of state and action $(s,a)$ is not seen in the training set. This is possible because when training the demonstration neural net, we always provide very good demonstration episodes. The demonstration network never sees imperfect data.
- **Ap20R2DFO**
    - Note: without mentioned, these experiments use r_scale=8 and r_shift=0
    - **Preselect**
        - **MeanVar**
            - _Result_: Converged. The uncertainty is highly non-deterministic, which makes sense because we select data every fixed points.
    - **Random**
        - **MeanVar**
            - _Result_: Converged. The uncertainty is highly non-deterministic, but are very small in general.
    - **Expert**
        - **MeanVarShift**
            - Added 0.5 r_shift
            - For the queries, I only captured the state within 0.2, so that we can look into the states near the goal state.
            - _Result_: The uncertainty looks not very ideal, but it is already greater than the random case
        - **MeanVarShift2**
            - Added 0.5 r_shift
            - For queries, sample everywhere to have a general overview
            - _Result_: The uncertainty looks not very ideal, but it is already greater than the random case
        - **NoInfer**
            - _Result_: Actually, when there is no r_shift, the estimated q is always greater than the demo 1, so the affect from demo q is small. We may still need to add some r_shift to the environment
        - **NoInferShift**
            - _Result_: To compare with the MeanVarShift2 result. See what the actual q should look like.

- **Ap02R2DFO**
    - Try to train the expert with different dataset and see the result, and the slightly reduce the dataset until it performs differently.
    - For this training, we always have a demonstration neural network, use direct update and counting.
    - Reward is also negative distance and we use scale = 12 and shift = 0.5
    - **T5000**: Using 5000 transitions.
        - **Expert**
            - *Result*: Did not converge when trained using demonstration network. The result is consistent with previous experiments. When checking the query.mp4, you can see that the demonstration neural network gives an over estimated Q when the input data is not seen in the dataset.
            - Note that for this expert training, I just used Q estimation from the pre-trained critic directly.
        - **Preselect**
            - Pre-select some points that covers the entire state and action spaces.
            - *Result*: worked fine. Just as using a trained critic function directly.
        - **Random**: Just randomly select some data within the observation and action space.
            - *Result*: worked fine. Similar to using a trained critic function.
    - **T5000**: Using 256 transitions only.
        - **Expert**
            - Same as T5000
        - **Preselect**
            - *Result*: According to the query.mp4, the output from the demonstration neural net is always lower than the output from the critic. So in this case, we did not really learn anything from the demonstration data.

- **Ap14R2DFO**
    - **QCompare**
        - Same environment configuration as above.
        - Train a ddpg environment used for generating output Q value.
        - Use a new generate rollouts to generate episodes for each selected (s,a) pair as the starting point.
        - **E100P4096**
            - _Result_: check the Q_compare.png figure in this directory. The real q value is very similar to the output from critic.
        - **E30P256**
            - _Result_: Similarly as above, but this time we take a closer look at the values.
    - **Uncertainty Check**
        - Same environment configuration as above.
        - The demonstration NN is trained with expert data only. Plot the uncertainty

## Reach2D
- **Ap07R2D**
    - We want to see if using a pre-trained critic network as the demonstrator is helpful for a slightly harder environment.
    - This environment is harder that the previous one as it is a second integrator and a multi-goal environment.
    - Still use negative reward with scale = 4 and shift = 0.5
    - Still use count and direct update rule
    - **CriticAsDemo**
        - **SingleGoal**
            - Use goal at (0,0). (Relaxed the threshold to boundary / 12)
        - **MultiGoal**
            - **Random**
                - Use 4096 transitions that are randomly generated (o, u, g are all randomly generated).
                - *Result*: Showed some improvement vs no demonstration.
            - **ExpertManual**
                - Use same number of transitions. The $q$ value is manually calculated for each episode.
                - *Result*: did not converge very well.
            - **ExpertAuto**
                - Use same number of transitions. The $q$ value is returned from a pre-trained critic.
                - *Result*: did not converge very well.

## FetchPush
- **Ap07FPush**
    - **CriticAsDemo**
        - *Result*: When using a pre-trained critic as demonstration nn, it converged very very fast!
        - Basically this means that DDPG can be easily bootstrapped.

## Ap19OpenAI
- train the DDPG with demonstration on the three OpenAI environemnts.
- **FetchPush**
- **FetchSlide**
- **FetchPickAndPlace**