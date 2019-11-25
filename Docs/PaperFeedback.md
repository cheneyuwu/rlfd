## Contribution:
- In order to understand the paper’s contribution, it would have helped to know that it is an application of the insight presented in [11] in combination with applying “recent” generative models such as GANs and normalizing flows.


## Related work
Suboptimal demonstrations are included by these RL approaches:
- “Overcoming Exploration in Reinforcement Learning with Demonstrations” by Nair et al.
- “Reinforcement Learning from Imperfect Demonstrations” by Gao et al.
- “Interactive Reinforcement Learning with Dynamic Reuse of Prior Knowledge from Human/Agent's Demonstration” by Wang et al.
- “Truncated Horizon Policy Search: Combining Reinforcement Learning & Imitation Learning” by Sun et al.
- “RL with supervision by a stable controller” by Rosenstein et al.
- “Residual RL for Robot Control” by Johannink et al.
- “Residual Policy Learning” by Silver et al.
    - Although one might argue that residual RL methods are irrelevant in this context, I think they are a reasonable alternative to deal with sub-optimal demonstrations and to avoid forgetting schedules:
- “Goal-conditioned Imitation Learning” Ding et al.
    - A few Imitation Learning approaches might be a more state-of-the-art comparison instead of pure Behavioral Cloning, e.g.:

- Pastor, Peter, et al. "Skill learning and task outcome prediction for manipulation." ICRA, 2011.
- Kormushev, Petar, at al. "Robot motor skill coordination with EM-based reinforcement learning." IROS, 2010.
- Kober, Jens, and Jan R. Peters. "Policy search for motor primitives in robotics." Advances in neural information processing systems. 2009.

These are more efficient than NN methods
- Chatzilygeroudis, Konstantinos, et al. "Black-box data-efficient policy search for robotics." IROS, 2017.
- Deisenroth, Marc, and Carl E. Rasmussen. "PILCO: A model-based and data-efficient approach to policy search." ICML, 2011.

Policy search methods
- Chatzilygeroudis et al. "A survey on policy search algorithms for learning robot controllers in a handful of trials"
- Deisenroth et al. "A Survey on Policy Search for Robotics"
    - Deep Learning methods are used solve problem that has been tackled with Policy Search Reinforcement Learning in a more data efficient fashion. Therefore, the convenient applicability of the approach to real robot systems is not fully evident. A deeper discussion and ideally a comparison is required on the advantage of deep-learning-based methods.


## Experiments
- What is the action space of the peg-insertion and pick-and-place task?
(From Fig. 3 it looks like a 2D action space?  I assume not, because in IV.B it is written that only(?) the sensitivity experiment is “limiting the state space to a 2D plane”.)
- Is the simulation purely kinematic or are forces considered?
- How is the picking action simulated? Does the end-effector only need to be in the vicinity of the object or do the fingers need to be actuated?
- How are the demonstrations generated?
- The argument against BC+\lambdaTD3 is its sensitivity to the hyperparameter \lambda (shown in Fig. 4). The proposed GAN shaping approach also contains hyperparameters, but no evaluation is shown. How sensitive is the proposed approach w.r.t. its hyperparameters?
- What is the tolerance in the peg-insertion task?
- Compare the performance of the proposed approach to the various methods presented in the past.

- Simplifications on real robots -> not easy
- What demonstrations are referred as sub-optimal?
- Replicate the real world experiment?
- Need more complicated environment for showing sub-optimality -> the 2D environment is not enough
- How many demonstrations are required in the algorithm?
- Shortest path metric is not the best metric for peg insertion

- Find some environments solvable by the baseline methods

- More complicated experiments

- Results in Section IV needs an introductory paragraph to describe the experimental concept and setup (data acquisition, data length both in simulation and real experiments, define the term “episodes” etc).
    - By now, figure plots of Section IV are not quite comprehensive, because of missing this description.


## Video
- It would be helpful to show the actual demonstrations.


## Typos & Expressions
- Overall, the paper is written clearly, a few comments:
    There is a confusing change of meaning of the lambda parameter. In Eq.
    2 lambda is a multiplier that relates to the demonstrations (higher
    lambda -> closer to demonstrations). In the experimental section lambda
    is a multiplier of TD3 (higher lambda -> closer to RL objective). 
- What is the meaning of green/red in Fig. 3 “Demonstrations”? (it helps to explain it in the caption)
- “It can handle multi-modal demonstrations gracefully” (p. 1): This might be misunderstood since multi-modal demonstrations oftentimes refer to a multimodal state space. Maybe re-use “multi-modal action distributions”.
- The explanations in sections IIIB and IIIC seem close to the original papers. Are they needed? Specify application-specific insights that were learned here.
- P. 2: “real” &#8594; “realm”
- P. 3: “since we assume kinesthetic teaching and not high-dimensional image data” &#8594; These two things are orthogonal.
- P. 5: “environment dos not” &#8594; “does not”  