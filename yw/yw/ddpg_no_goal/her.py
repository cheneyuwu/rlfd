import numpy as np


def make_sample_transitions():
    def _sample_transitions(episode_batch, batch_size_in_transitions):
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
        
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        assert transitions["u"].shape[0] == batch_size_in_transitions

        return transitions

    return _sample_transitions