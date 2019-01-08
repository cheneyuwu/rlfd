# [DDPG Algorithm](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
## Key Notes and Equations
  - Q learning side
    - Input: state/observation $s$, goal of this episode $g$ and action $a$
    - Bellman equation
      - $Q^*(s,a)=\underset{s' \sim P}{E}[r(s,a)+\gamma \underset{a'}{max}Q^*(s',a')]$
      - $s'$ and $a'$ are the next state and action, respectively.
    - Mean-squared Bellman error (MSBE)
      - $L(\phi, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
          \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a') \right) \Bigg)^2
          \right]$
      - $d$ to indicate whether $s'$ is a terminal state (not used in my implementation)
    - Trick 1: replay buffer that stores tuple of $(s,a,r,s',d)$
    - Trick 2: target networks
      - Target: $r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a')$
      - Polyas averaging: $\phi_{\text{targ}} \leftarrow \rho \phi_{\text{targ}} + (1 - \rho) \phi$
    - Loss function:
      - $L(\phi, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
        \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) Q_{\phi_{\text{targ}}}(s', \mu_{\theta_{\text{targ}}}(s')) \right) \Bigg)^2
        \right]$
  - Policy learning side, gradient ascent to maximize Q value.
    - $\max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi}(s, \mu_{\theta}(s)) \right]$
  - Exploration vs. Exploitation
    - $\epsilon$-greedy
    - uncorrelated, mean-zero Gaussian noise
## Related Papers