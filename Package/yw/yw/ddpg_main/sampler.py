import numpy as np


def make_sample_her_transitions(strategy, k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        strategy (str) - set to "future" to use the HER replay strategy; if set to 'none', regular DDPG experience replay is used
        k        (int) - the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if strategy == "future":
        future_p = 1 - (1.0 / (1 + k))
    else:  # 'strategy' == 'none'
        future_p = 0

    def sample_her_transitions(episode_batch, batch_size_in_transitions):
        """
        episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch["u"].shape[1]
        rollout_batch_size = episode_batch["u"].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch["ag"][episode_idxs[her_indexes], future_t]
        transitions["g"][her_indexes] = future_ag

        if "q" in transitions.keys():
            transitions["q"][her_indexes] = -100 * np.ones(transitions["q"][her_indexes].shape)

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith("info_"):
                info[key.replace("info_", "")] = value
        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ["ag_2", "g"]}
        reward_params["info"] = info
        transitions["r"] = reward_fun(**reward_params).reshape(-1, 1)  # reshape to be consistent with default reward

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        assert transitions["u"].shape[0] == batch_size_in_transitions

        return transitions

    return sample_her_transitions


def make_sample_nstep_transitions(gamma, n):
    """Creates a sample function that can be used for n step return

    Args:
        gamma (str) - discount rate
        n        (int) - the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times as many HER replays as regular replays are used)
    """
    def sample_nstep_transitions(episode_batch, batch_size):
        T = episode_batch["u"].shape[1]
        total_batch_size = episode_batch["u"].shape[0]

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, total_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        
        # Get the transitions
        transitions = {}
        for k in ["o", "u", "g", "ag", "info_is_success", "q", "mask"]:
            transitions[k] = episode_batch[k][episode_idxs, t_samples].copy()
        # calculate n step return
        cum_reward = np.zeros_like(episode_batch["r"][episode_idxs, t_samples])
        cum_discount = np.zeros_like(episode_batch["n"][episode_idxs, t_samples])
        assert n>= 1
        for step in range(n):
            cum_reward += np.where(((t_samples + step) < T).reshape(cum_reward.shape), episode_batch["r"][episode_idxs, np.minimum(T-1, t_samples + step)] * np.power(gamma, step), 0)
            cum_discount += np.where(((t_samples + step) < T).reshape(cum_discount.shape), 1, 0)
        transitions["r"] = cum_reward
        transitions["n"] = cum_discount
        # change the state it goes to
        n_step_t_samples = np.minimum(t_samples + n - 1, T-1)
        for k in ["o_2", "ag_2"]:
            transitions[k] = episode_batch[k][episode_idxs, n_step_t_samples].copy()
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        assert transitions.keys() == episode_batch.keys()
        assert transitions["u"].shape[0] == batch_size

        return transitions
    
    return sample_nstep_transitions
    
