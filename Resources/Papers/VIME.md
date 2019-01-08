# [VIME]
## Dependency
- Implemented on model based policy gradient
## Key Notes and Equations
- $P_\alpha(\tau)=P(s_0)\Pi P(s_{t+1}|s_t ,a_t ,\theta)\Pi \pi(a_t|s_t,\alpha)$
- Using DDPG, policy $\pi$ is deterministic so that the only non-deterministism happens in state trasition. VIME tries to choose a policy such that the transition probability ($s_t, a_t, s_{t+1}$) are most informative for training the dynamic model.
- State transition model: $\int p(s_{t+1}|s_t, a_t; \theta)p(\theta)$
- Maximizing the sum of reductions in entropy (uncertainty)
  - $\sum_t(H(\Theta|\xi_t,a_t)-H(\Theta|S_{t+1},a_t))=\sum_tI(S_{t+1};\Theta|\xi_t,a_t)$
  - $\sum_tI(S_{t+1};\Theta|\xi_t,a_t)=E_{s_{t+1}~P(\centerdot | \xi_t, a_t)}[D_{KL}[p(\theta|\xi_t,a_t,s_{t+1})||p(\theta|\xi_t)]]$
- Bayes' rule
  - $p(\theta|\xi_t,a_t,s_{t+1})=\dfrac{p(\theta|\xi_t)p(s_{t+1}|\xi_t,a_t;\theta)}{p(s_{t+1}|\xi_t,a_t)}$
  - This is difficult to compute in high dimensional space.
- Variational inference
  - Approximate $p(\theta|D)$ through an alternative distribution $q(\theta;\phi)$, parameterized by $\phi$, and minimize $D_{KL}[q(\theta;\phi)||p(\theta|D)]$.
  - Maximize the variational lower bound $L[q(\theta;\phi)||p(\theta)]$
    - $L[q(\theta;\phi)||p(\theta)]=E_{\theta\sim q(\centerdot;\phi)}[logp(D|\theta)]-D_{KL}[q(\theta;\phi)||p(\theta)]$
    - The first term is log likelihood
    - The second term is distance to **prior**
- Update rules under the config of Gaussian distribution
  - Model - BNN 
    - $q(\theta; \phi)=\Pi_{i=1}^{|\Theta|}\N(\theta_i|\mu_i;\sigma_i^2)$ with $\phi=\{\mu,\sigma\}$
    - Also use $\sigma=log(1+e^\rho)$
  - Maximize variational lower bound $L[q(\theta;\phi)||p(\theta)]$
    - $\phi' =\underset{\phi}{arg\ min}[D_{KL}[q(\theta;\phi||q(\theta;\phi_{t-1}))]-E_{\theta \sim q(\centerdot ; \phi)}[log\ p(s_t|\xi_t,a_t;\theta)]]$
    - Note 1: $E_{\theta\sim q(\centerdot;\phi)}[logp(D|\theta)] \approx \dfrac{1}{N}\sum ^N_{i=1}logp(D|\theta_i)$ with $N$ samples drawn according to $\theta \sim q(\centerdot ; \phi)$
  - Compute $D_{KL}[q(\theta;\phi')||q(\theta;\phi)]$ by approximating $\nabla^TH^{-1}\nabla$ of the above for each $(s_t,a_t,s_{t+1})$ tuple generated during rollout.
- Modified reward for policy:
    - $r\prime(s_t,a_t,s_{t+1})=r(s_t,a_t)+\eta D_{KL}[q(\theta;\phi_{t+1}||q(\theta;\phi_t))]$, where $\phi_{t+1}$ is the prior belief, $\phi_t$ is the posterior reward.
## Notes from Paper
2 Methodology  
2.2 Curiosity  
- The state transition model
  - Integral from infinity of $p(s_{t+1}|s_t, a_t; \theta)p(\theta)$
- The agent should take actions that maximize the reduction in uncertainty about the dynamics, which is equivalent to maximizing the sum of reductions in entropy
  - $\sum_t(H(\Theta|\xi_t,a_t)-H(\Theta|S_{t+1},a_t))=\sum_tI(S_{t+1};\Theta|\xi_t,a_t)$
  - $\sum_tI(S_{t+1};\Theta|\xi_t,a_t)=E_{s_{t+1}~P(\centerdot | \xi_t, a_t)}[D_{KL}[p(\theta|\xi_t,a_t,s_{t+1})||p(\theta|\xi_t)]]$
  - 
2.3 Variational Bayes  
- Bayes' rule
  - $p(\theta|\xi_t,a_t,s_{t+1})=\dfrac{p(\theta|\xi_t)p(s_{t+1}|\xi_t,a_t;\theta)}{p(s_{t+1}|\xi_t,a_t)}$
- Variational inference
  - Approximate $p(\theta|D)$ through an alternative distribution $q(\theta;\phi)$, parametrized by $\phi$, and minimize $D_{KL}[q(\theta;\phi)||p(\theta|D)]$
  - Maximize the variational lower bound $L[q(\theta;\phi)||p(\theta)]$
    - $L[q(\theta;\phi)||p(\theta)]=E_{\theta~q(\centerdot;\phi)}[logp(D|\theta)]-D_{KL}[q(\theta;\phi)||p(\theta)]$
  - After reparametrization, the final reward is then:
    - $r\prime(s_t,a_t,s_{t+1})=r(s_t,a_t)+\eta D_{KL}[q(\theta;\phi_{t+1}||q(\theta;\phi_t))]$
    - 
2.5 Implementation  
  - BNN 
    - $q(\theta; \phi)=\Pi_{i=1}^{|\Theta|}\N(\theta_i|\mu_i;\sigma_i^2)$ with $\phi=\{\mu,\sigma\}$
    - Also use $\sigma=log(1+e^\rho)$
    - $E_{\theta~q(\centerdot;\phi)} \approx \dfrac{1}{N}\sum ^N_{i=1}logp(D|\theta_i)$ with $N$ samples drawn according to $\theta \sim q(\centerdot ; \phi)$
    - The posterior distribution of the dynamics parameter (to update $\phi$), i.e. variational lower bound
      - $\phi \prime =arg min_{\phi}[D_{KL}[q(\theta;\phi||q(\theta;\phi_{t-1}))]-E_{\theta \sim q(\centerdot ; \phi)}[log p(s_t|\xi_t,a_t;\theta)]]$
      - $\Delta \phi=H^{-1}(l)\nabla_\phi l(q(\theta;\phi),s_t)$
      - Take the advantage of Gaussian: 
        - $D_{KL}[q(\theta;\phi)||p(\theta|\phi\prime)]=\dfrac{1}{2}\sum{|\Theta|}{i=1}()$
## Check Later
- epi-greedy or Boltzmann exploration [7]
- utilizing Gaussian noise on the controls in policy gradient methods [8]
- curiosity-driven exploration [16, 17, 21, 22]
- Optimizing the variational lower bound in combination with the reparametrization trick is called stochastic gradient variational Bayes (SGVB) [26]. Sampling at the weights is replaced by sampling the neuron pre-activations
## Question
- Assume that the state transition is non-deterministic, how can reward function be deterministic?
