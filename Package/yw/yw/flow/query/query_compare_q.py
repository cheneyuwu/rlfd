"""This is a very dangerous script. It copies too many existing functions and modified them a little bit just to
do the job. Remember to ensure that these function are in sync everytime you use this script.
"""

import click
import os
import numpy as np
import pickle

import matplotlib

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import matplotlib.patches as mpatches
# import matplotlib.animation as animation


# DDPG Package import
from yw.tool import logger
from yw.ddpg_main import config
from yw.util.mpi_util import set_global_seeds
from yw.util.lit_util import toy_regression_data
from yw.util.util import store_args


from mujoco_py import MujocoException   # temporarily disable this because no mujoco on mac
from collections import deque



class RolloutWorker:
    @store_args
    def __init__(
        self,
        make_env,
        policy,
        dims,
        T,
        rollout_batch_size=1,
        compute_r=False,
        exploit=False,
        use_target_net=False,
        compute_Q=False,
        noise_eps=0,
        random_eps=0,
        history_len=100,
        render=False,
        **kwargs
    ):
        """
        Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env           (function)    - a factory function that creates a new instance of the environment when called
            policy             (object)      - the policy that is used to act
            dims               (dict of int) - the dimensions for observations (o), goals (g), and actions (u)
            rollout_batch_size (int)         - the number of parallel rollouts that should be used
            exploit            (bool)        - whether or not to exploit, i.e. to act optimally according to the current policy without any exploration
            use_target_net     (bool)        - whether or not to use the target net for rollouts
            compute_Q          (bool)        - whether or not to compute the Q values alongside the actions
            noise_eps          (float)       - scale of the additive Gaussian noise
            random_eps         (float)       - probability of selecting a completely random action
            history_len        (int)         - length of history for statistics smoothing
            render             (bool)        - whether or not to render the rollouts
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0  # time horizon

        self.info_keys = [key.replace("info_", "") for key in dims.keys() if key.startswith("info_")]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims["g"]), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims["o"]), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims["g"]), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        self.initial_o[i] = obs["observation"]
        self.initial_ag[i] = obs["achieved_goal"]
        self.g[i] = obs["desired_goal"]

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self, initial=None):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims["o"]), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims["g"]), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, rewards, successes = [], [], [], [], [], []
        Qs = []
        info_values = [
            np.empty((self.T, self.rollout_batch_size, self.dims["info_" + key]), np.float32) for key in self.info_keys
        ]

        if initial is not None:
            self.initial_o = initial["o"]
            self.initial_ag = initial["o"]
            o[:] = self.initial_o
            ag[:] = self.initial_ag
            u = initial["u"]
            Q = np.zeros((self.rollout_batch_size, 1))
            o_new = np.empty((self.rollout_batch_size, self.dims["o"]))
            ag_new = np.empty((self.rollout_batch_size, self.dims["g"]))
            r = np.empty((self.rollout_batch_size, 1)) # reward
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                new_obs = self.envs[i].reset(init = {"observation": self.initial_o[i], "goal": initial["g"][i]})
                self.g[i] = new_obs["desired_goal"]
                if self.render:
                    self.envs[i].render()
                curr_o_new, curr_r, _, info = self.envs[i].step(u[i])
                r[i] = curr_r
                if "is_success" in info:
                    success[i] = info["is_success"]
                o_new[i] = curr_o_new["observation"]
                ag_new[i] = curr_o_new["achieved_goal"]
                for idx, key in enumerate(self.info_keys):
                    info_values[idx][0, i] = info[key]
                if self.render:
                    self.envs[i].render()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            rewards.append(r.copy())
            if self.compute_Q:
                Qs.append(Q.copy())
            o[...] = o_new
            ag[...] = ag_new


        for t in range(self.T - int(initial != None)):
            logger.debug("RolloutWorker.generate_rollouts -> step {}".format(t))
            policy_output = self.policy.get_actions(
                o,
                ag,
                self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.0,
                random_eps=self.random_eps if not self.exploit else 0.0,
                use_target_net=self.use_target_net,
            )
            if self.compute_Q:
                u = policy_output[0]
                Q = policy_output[1]
                Q = Q.reshape(-1,1)
            else:
                u = np.array(policy_output)
            u = u.reshape(-1, self.dims["u"])

            o_new = np.empty((self.rollout_batch_size, self.dims["o"]))
            ag_new = np.empty((self.rollout_batch_size, self.dims["g"]))
            r = np.empty((self.rollout_batch_size, 1)) # reward
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                # Method 1
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, curr_r, _, info = self.envs[i].step(u[i])
                    r[i] = curr_r
                    if "is_success" in info:
                        success[i] = info["is_success"]
                    o_new[i] = curr_o_new["observation"]
                    ag_new[i] = curr_o_new["achieved_goal"]
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                except MujocoException as e:
                    return self.generate_rollouts()
                # Method 2
                # curr_o_new, curr_r, _, info = self.envs[i].step(u[i])
                # r[i] = curr_r
                # if "is_success" in info:
                #     success[i] = info["is_success"]
                # o_new[i] = curr_o_new["observation"]
                # ag_new[i] = curr_o_new["achieved_goal"]
                # for idx, key in enumerate(self.info_keys):
                #     info_values[idx][t, i] = info[key]
                # if self.render:
                #     self.envs[i].render()

            if np.isnan(o_new).any():
                logger.warn("NaN caught during rollout generation. Trying again...")
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            rewards.append(r.copy())
            if self.compute_Q:
                Qs.append(Q.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())

        # Will contain episode info
        episode = dict(o=obs, u=acts, g=goals, ag=achieved_goals)
        if self.compute_r:
            episode["r"] = rewards
        if self.compute_Q:
            episode["q"] = Qs
        for key, value in zip(self.info_keys, info_values):
            episode["info_{}".format(key)] = value
        episode = RolloutWorker.convert_episode_to_batch_major(episode)

        # Stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return episode

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """
        Pickles the current policy for later inspection.
        """
        with open(path, "wb") as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix="worker"):
        """
        Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [("success_rate", np.mean(self.success_history))]
        if self.compute_Q:
            logs += [("mean_Q", np.mean(self.Q_history))]
        logs += [("episode", self.n_episodes)]

        if prefix is not "" and not prefix.endswith("/"):
            return [(prefix + "/" + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)

    @staticmethod
    def convert_episode_to_batch_major(episode):
        """Converts an episode to have the batch dimension in the major (first)
        dimension.
        """
        episode_batch = {}
        for key in episode.keys():
            val = np.array(episode[key]).copy()
            # make inputs batch-major instead of time-major
            episode_batch[key] = val.swapaxes(0, 1)
        return episode_batch



def generate_demo_data(policy_file, query_file, store_dir):
    """
    Generate demo from policy file
    """
    assert policy_file is not None, "Must provide the policy_file!"
    set_global_seeds(0)

    # Load random data from npz file.
    initial = {**np.load(query_file)}

    # Load policy.
    with open(policy_file, "rb") as f:
        policy = pickle.load(f)

    # Extract environment construction information
    env_name = policy.info["env_name"]
    r_scale = policy.info["r_scale"]
    r_shift = policy.info["r_shift"]
    eps_length = policy.info["eps_length"]

    # Prepare params.
    params = {}
    params["env_name"] = env_name
    params["r_scale"] = r_scale
    params["r_shift"] = r_shift
    params["eps_length"] = eps_length
    params["rank_seed"] = 0
    params["render"] = False
    params["rollout_batch_size"] = initial["o"].shape[0]
    params = config.add_env_params(params=params)
    demo_params = {
        "exploit": True,
        "use_target_net": True,
        "use_demo_states": False,
        "compute_Q": False,
        "compute_r": True,
        "render": params["render"],
        "T": policy.T,
        "rollout_batch_size": params["rollout_batch_size"],
        "dims": params["dims"],
    }
    demo = RolloutWorker(params["make_env"], policy, **demo_params)
    demo.seed(params["rank_seed"])

    # Run evaluation.
    demo.clear_history()
    episode = demo.generate_rollouts(initial)

    # record logs
    for key, val in demo.logs("test"):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()

    # add expected Q value
    exp_q = np.empty(episode["r"].shape)
    exp_q[:, -1, :] = episode["r"][:, -1, :] / (1 - policy.gamma)
    for i in range(policy.T - 1):
        exp_q[:, -2 - i, :] = policy.gamma * exp_q[:, -1 - i, :] + episode["r"][:, -2 - i, :]
    episode["q"] = exp_q

    initial["q_manual"] = episode["q"][:, 0, ...].reshape(-1, 1)

    # Plot the result
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.linspace(-1.0, 1.0, params["rollout_batch_size"])
    ax.plot(x, initial["q_manual"].reshape((-1)), label="Manual")
    ax.plot(x, initial["q"].reshape((-1)), label="Critic")
    ax.legend()
    ax.set_title("Critic vs Demonstration NN Output")
    ax.set_xlabel("Flattened state action space")
    ax.set_ylabel("Estimated Q value")
    plt.show(block=False)
    plt.pause(2)
    save_path = os.path.join(store_dir, "Q_compare.png")
    plt.savefig(save_path)
    print("Save image to "+save_path)


@click.command()
@click.option("--policy_file", type=str, default=None, help="Input policy for training.")
@click.option("--query_file", type=str, default=None, help="The query file that contains initial state and action pairs.")
@click.option("--store_dir", type=str, default=None, help="The query file that contains initial state and action pairs.")
def main(**kwargs):
    logger.set_level(2)
    generate_demo_data(**kwargs)

if __name__ == "__main__":
    main()

