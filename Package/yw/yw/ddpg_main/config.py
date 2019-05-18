import numpy as np

from yw.tool import logger
from yw.ddpg_main.ddpg import DDPG
from yw.ddpg_main.imitation import QEstimator, ActionImitator
from yw.ddpg_main.gp_imitation import GPQEstimator
from yw.ddpg_main.rollout import RolloutWorker
from yw.ddpg_main.her import make_sample_her_transitions

from yw.env.env_manager import EnvManager


DEBUG_PARAMS = {
    "rl_layers": 2,  # number of layers in the critic/actor networks
    "rl_hidden": 4,  # number of neurons in each hidden layers
    "demo_layers": 2,  # number of layers in the demo networks
    "demo_hidden": 4,  # number of neurons in each hidden layers
    "demo_data_size": 16,  # training data size in number of transitions
    "demo_batch_size": 4,  # number of samples to be used to train the demo nn
    "demo_test_batch_size": 4,  # testing data size in number of transitions
    "n_cycles": 2,  # per epoch
    "n_batches": 2,  # training batches per cycle
    "rl_batch_size": 4,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    "rollout_batch_size": 2,  # per mpi thread
    "n_test_rollouts": 2,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
}

DEFAULT_PARAMS = {
    # config name
    "config": "default",
    # env
    "env_name": "FetchPickAndPlace-v1",
    "r_scale": 1.0,  # re-scale the reward. Only use this for dense rewards.
    "r_shift": 0.0,  # re-scale the reward. Only use this for dense rewards.
    "eps_length": 0,  # change the length of the episode.
    "max_u": 1.0,  # max absolute value of actions on different coordinates
    "no_pos_return": False,  # Whether or not this environment has positive return or not.
    # general ddpg config
    "buffer_size": int(1e6),  # for experience replay
    "rl_scope": "ddpg",  # can be tweaked for testing
    "rl_ca_ratio": 1,  # ratio of critic over actor, 1 -> ddpg, 2 -> td3
    "rl_num_sample": 1,  # number of ensemble of actor_critics
    "rl_layers": 3,  # number of layers in the critic/actor networks
    "rl_hidden": 256,  # number of neurons in each hidden layers
    "rl_Q_lr": 0.001,  # critic learning rate
    "rl_pi_lr": 0.001,  # actor learning rate
    "rl_action_l2": 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    "rl_batch_size": 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    "rl_batch_size_demo": 128,  # number of samples to be used from the demonstrations buffer, per mpi thread 128/1024 or 32/256
    "rl_demo_critic": "none",  # whether or not to use the behavior cloning loss as an auxilliary loss
    "rl_demo_actor": "none",  # whether or not to use the behavior cloning loss as an auxilliary loss
    "rl_q_filter": 0,  # whether or not a Q value filter should be used on the Actor outputs
    "rl_num_demo": 1000,  # number of expert demo episodes
    "rl_prm_loss_weight": 0.001,  # Weight corresponding to the primary loss
    "rl_aux_loss_weight": 0.0078,  # Weight corresponding to the auxilliary loss also called the cloning loss
    # Double Q Learning
    "rl_polyak": 0.95,  # polyak averaging coefficient
    # Normalization - not used for now due to usage of demonstration
    "rl_norm_eps": 0.01,  # epsilon used for observation normalization
    "rl_norm_clip": 5,  # normalized observations are cropped to this values
    # Process I/Os
    "relative_goals": False,
    "clip_obs": 200.0,
    "clip_return": True,
    # Exploration - for rollouts
    "exploit": False,  # whether or not to use e-greedy and add noise to output
    "random_eps": 0.3,  # percentage of time a random action is taken
    "noise_eps": 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    "replay_strategy": "future",  # supported modes: future, none
    "replay_k": 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # use demonstration to shape actor or critic
    "demo_scope": "demo",
    "demo_net_type": "ensemble",  # choose between "ensemble" or "baysian"
    "demo_num_sample": 10,  # number of bootstrapped ensemble of demonstration neural nets.
    "demo_layers": 3,  # number of layers in the demo networks
    "demo_hidden": 256,  # number of neurons in each hidden layers
    "demo_lr": 0.001,  # demonstration nn learning rate.
    # Demonstration training
    "train_demo_epochs": 1,
    "demo_data_size": 4096,  # training data size in number of transitions
    "demo_batch_size": 128,  # number of samples to be used to train the demo nn
    "demo_test_batch_size": 100,  # testing data size in number of transitions
    # DDPG Training
    "train_rl_epochs": 1,
    "n_cycles": 50,  # per epoch
    "n_batches": 40,  # training batches per cycle
    "rollout_batch_size": 4,  # per mpi thread
    "n_test_rollouts": 20,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    # Testing
    "test_with_polyak": False,  # run test episodes with the target network
}


def log_params(params):
    for key in sorted(params.keys()):
        logger.info("{}: {}".format(key, params[key]))


# Helper Functions for Configuration
# =====================================


class EnvCache:
    """Only creates a new environment from the provided function if one has not yet already been
    created.

    This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.

    """

    cached_envs = {}

    @staticmethod
    def get_env(make_env):
        if make_env not in EnvCache.cached_envs.keys():
            EnvCache.cached_envs[make_env] = make_env()
        return EnvCache.cached_envs[make_env]


def add_env_params(params):
    env_manager = EnvManager(params["env_name"], params["r_scale"], params["r_shift"], params["eps_length"])
    logger.info(
        "Using environment %s with r scale down by %f shift by %f and max episode %f"
        % (params["env_name"], params["r_scale"], params["r_shift"], params["eps_length"])
    )
    params["make_env"] = env_manager.get_env
    tmp_env = EnvCache.get_env(params["make_env"])
    assert hasattr(tmp_env, "_max_episode_steps")
    params["T"] = tmp_env._max_episode_steps
    params["gamma"] = 1.0 - 1.0 / params["T"]
    assert hasattr(tmp_env, "max_u")
    params["max_u"] = np.array(tmp_env.max_u) if isinstance(tmp_env.max_u, list) else tmp_env.max_u
    # get environment dimensions
    tmp_env.reset()
    obs, _, _, info = tmp_env.step(tmp_env.action_space.sample())
    dims = {"o": obs["observation"].shape[0], "u": tmp_env.action_space.shape[0], "g": obs["desired_goal"].shape[0]}
    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims["info_{}".format(key)] = value.shape[0]
    params["dims"] = dims
    return params


def extract_params(params, prefix=""):
    extracted_params = {key.replace(prefix, ""): params[key] for key in params.keys() if key.startswith(prefix)}
    for key in extracted_params.keys():
        params["_" + key] = params[prefix + key]
        del params[prefix + key]
    return extracted_params


def configure_her(params):
    env = EnvCache.get_env(params["make_env"])
    env.reset()

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    her_params = extract_params(params, "replay_")
    her_params["reward_fun"] = reward_fun
    logger.info("*** her_params ***")
    log_params(her_params)
    logger.info("*** her_params ***")
    sample_her_transitions = make_sample_her_transitions(**her_params)

    return sample_her_transitions


def configure_ddpg(params, demo_policy):
    # Extract relevant parameters.
    ddpg_params = extract_params(params, "rl_")
    for name in ["max_u", "buffer_size", "relative_goals", "clip_obs"]:
        ddpg_params[name] = params[name]
        params["_ddpg_" + name] = params[name]
        del params[name]

    sample_her_transitions = configure_her(params)
    # Update parameters
    ddpg_params.update(
        {
            "input_dims": params["dims"].copy(),  # agent takes an input observations
            "T": params["T"],
            "clip_pos_returns": params["no_pos_return"],  # clip positive returns
            "clip_return": (1.0 / (1.0 - params["gamma"])) if params["clip_return"] else np.inf,  # max abs of return
            "rollout_batch_size": params["rollout_batch_size"],
            "subtract_goals": simple_goal_subtract,
            "sample_transitions": sample_her_transitions,
            "gamma": params["gamma"],
        }
    )
    ddpg_params["info"] = {
        "env_name": params["env_name"],
        "r_scale": params["r_scale"],
        "r_shift": params["r_shift"],
        "eps_length": params["eps_length"],
    }
    logger.info("*** ddpg_params ***")
    log_params(ddpg_params)
    logger.info("*** ddpg_params ***")
    policy = DDPG(demo_policy=demo_policy, **ddpg_params)
    return policy


def configure_nn_q_estimator(params):

    # Extract relevant parameters.
    demo_params = extract_params(params, "demo_")
    for name in ["T", "max_u", "buffer_size"]:
        # copy public parameters
        demo_params[name] = params[name]
    demo_params["input_dims"] = params["dims"].copy()

    logger.info("*** nn_q_estimator_params ***")
    log_params(demo_params)
    logger.info("*** nn_q_estimator_params ***")
    policy = QEstimator(**demo_params)

    return policy

def configure_gp_q_estimator(params):

    # Extract relevant parameters.
    demo_params = extract_params(params, "demo_")
    for name in ["T", "max_u", "buffer_size"]:
        # copy public parameters
        demo_params[name] = params[name]
    demo_params["input_dims"] = params["dims"].copy()

    logger.info("*** gp_q_estimator_params ***")
    log_params(demo_params)
    logger.info("*** gp_q_estimator_params ***")
    policy = GPQEstimator(**demo_params)

    return policy


def configure_action_imitator(params):

    # Extract relevant parameters.
    demo_params = extract_params(params, "demo_")
    for name in ["T", "max_u", "buffer_size"]:
        # copy public parameters
        demo_params[name] = params[name]
    demo_params["input_dims"] = params["dims"].copy()
    demo_params["info"] = {
        "env_name": params["env_name"],
        "r_scale": params["r_scale"],
        "r_shift": params["r_shift"],
        "eps_length": params["eps_length"],
    }

    logger.info("*** action_imitator_params ***")
    log_params(demo_params)
    logger.info("*** action_imitator_params ***")
    policy = ActionImitator(**demo_params)

    return policy


def config_rollout(params, policy):
    rollout_params = {
        "exploit": params["exploit"],
        "use_target_net": False,
        "use_demo_states": True,
        "compute_Q": False,
        "T": params["T"],
    }
    for name in ["rollout_batch_size", "dims", "noise_eps", "random_eps"]:
        rollout_params[name] = params[name]

    logger.info("\n*** rollout_params ***")
    log_params(rollout_params)
    logger.info("*** rollout_params ***")
    rollout_worker = RolloutWorker(params["make_env"], policy, **rollout_params)
    rollout_worker.seed(params["rank_seed"])
    return rollout_worker


def config_evaluator(params, policy):
    eval_params = {
        "exploit": 1,
        "use_target_net": params["test_with_polyak"],
        "use_demo_states": False,
        "compute_Q": True,
        "T": params["T"],
        "rollout_batch_size": params["n_test_rollouts"],
    }
    for name in ["dims", "noise_eps", "random_eps"]:
        eval_params[name] = params[name]
    logger.info("*** eval_params ***")
    log_params(eval_params)
    logger.info("*** eval_params ***")
    evaluator = RolloutWorker(params["make_env"], policy, **eval_params)
    evaluator.seed(params["rank_seed"])
    return evaluator


def config_demo(params, policy):
    demo_params = {
        "exploit": True,
        "use_target_net": True,
        "use_demo_states": False,
        "compute_Q": True,
        "compute_r": True,
        "render": params["render"],
        "T": policy.T,
        "rollout_batch_size": params["rollout_batch_size"],
        "dims": params["dims"],
    }
    logger.info("*** demo_params ***")
    log_params(demo_params)
    logger.info("*** demo_params ***")
    demo = RolloutWorker(params["make_env"], policy, **demo_params)
    demo.seed(params["rank_seed"])
    return demo


# Other helper functions
# =====================================


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b
