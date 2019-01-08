# =============================================================================
# Import
# =============================================================================

# System import
import numpy as np
import gym

# DDPG Package import
from yw import logger
from yw.td3.ddpg import DDPG
from yw.ddpg.rollout import RolloutWorker
from yw.ddpg.her import make_sample_her_transitions

DEFAULT_PARAMS = {
    # config name
    "config": "td3",
    # env
    "max_u": 1.0,  # max absolute value of actions on different coordinates
    # ddpg
    "layers": 3,  # number of layers in the critic/actor networks
    "hidden": 256,  # number of neurons in each hidden layers
    "network_class": "yw.td3.actor_critic:ActorCritic",
    "Q_lr": 0.001,  # critic learning rate
    "pi_lr": 0.001,  # actor learning rate
    "buffer_size": int(1e6),  # for experience replay
    "polyak": 0.95,  # polyak averaging coefficient
    "action_l2": 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    "clip_obs": 200.0,
    "scope": "ddpg",  # can be tweaked for testing
    "relative_goals": False,
    # Training
    "n_cycles": 10,  # per epoch
    "rollout_batch_size": 5,  # per mpi thread
    "n_batches": 40,  # training batches per cycle
    "batch_size": 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    "n_test_rollouts": 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    # Exploration
    "random_eps": 0.3,  # percentage of time a random action is taken
    "noise_eps": 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    "replay_strategy": "none",  # supported modes: future, none
    "replay_k": 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # Normalization
    "norm_eps": 0.01,  # epsilon used for observation normalization
    "norm_clip": 5,  # normalized observations are cropped to this values
    # Demonstration
    # 'bc_loss': 0, # whether or not to use the behavior cloning loss as an auxilliary loss
    # 'q_filter': 0, # whether or not a Q value filter should be used on the Actor outputs
    # 'num_demo': 100, # number of expert demo episodes
    # 'demo_batch_size': 128, #number of samples to be used from the demonstrations buffer, per mpi thread 128/1024 or 32/256
    # 'prm_loss_weight': 0.001, #Weight corresponding to the primary loss
    # 'aux_loss_weight':  0.0078, #Weight corresponding to the auxilliary loss also called the cloning loss
    # Test
    'test_with_polyak': False,  # run test episodes with the target network
}

CACHED_ENVS = {}


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        print("create new environment")
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def configure_ddpg(params, reuse=False, use_mpi=True, clip_return=True):
    # Extract relevant parameters.
    ddpg_params = dict()
    for name in [
        "max_u",
        "hidden",
        "layers",
        "batch_size",
        "network_class",
        "polyak",
        "buffer_size",
        "Q_lr",
        "pi_lr",
        "relative_goals",
        "norm_eps",
        "norm_clip",
        "clip_obs",
        "action_l2",
        "scope",
    ]:
        ddpg_params[name] = params[name]
        params["_" + name] = params[name]
        del params[name]
    # Extract relevant parameters.
    gamma = params["gamma"]
    rollout_batch_size = params["rollout_batch_size"]
    sample_her_transitions = configure_her(params)
    # Update parameters
    ddpg_params.update(
        {
            "input_dims": params["dims"].copy(),  # agent takes an input observations
            "T": params["T"],
            "clip_pos_returns": True,  # clip positive returns
            "clip_return": (1.0 / (1.0 - gamma)) if params["clip_return"] else np.inf,  # max abs of return
            "rollout_batch_size": rollout_batch_size,
            "subtract_goals": simple_goal_subtract,
            "sample_transitions": sample_her_transitions,
            "gamma": gamma,
            # 'bc_loss': params['bc_loss'],
            # 'q_filter': params['q_filter'],
            # 'num_demo': params['num_demo'],
            # 'demo_batch_size': params['demo_batch_size'],
            # 'prm_loss_weight': params['prm_loss_weight'],
            # 'aux_loss_weight': params['aux_loss_weight'],
        }
    )
    ddpg_params["info"] = {"env_name": params["env_name"]}
    logger.info("\n*** ddpg_params ***")
    log_params(ddpg_params)
    policy = DDPG(**ddpg_params)
    return policy


def configure_dims(params):  # this is how the cached info is used for.
    env = cached_make_env(params["make_env"])
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())

    dims = {"o": obs["observation"].shape[0], "u": env.action_space.shape[0], "g": obs["desired_goal"].shape[0]}
    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims["info_{}".format(key)] = value.shape[0]
    return dims


def config_rollout(params, policy):
    rollout_params = {
        "exploit": False,
        "use_target_net": False,
        "use_demo_states": True,
        "compute_Q": False,
        "T": params["T"],
    }
    for name in ["T", "rollout_batch_size", "dims", "noise_eps", "random_eps"]:
        rollout_params[name] = params[name]
    logger.info("\n*** rollout_params ***")
    log_params(rollout_params)
    rollout_worker = RolloutWorker(params["make_env"], policy, **rollout_params)
    rollout_worker.seed(params["rank_seed"])
    return rollout_worker


def config_evaluator(params, policy):
    eval_params = {
        "exploit": True,
        "use_target_net": params['test_with_polyak'],
        "use_demo_states": False,
        "compute_Q": True,
        "T": params["T"],
    }
    for name in ["T", "rollout_batch_size", "dims", "noise_eps", "random_eps"]:
        eval_params[name] = params[name]
    logger.info("\n*** eval_params ***")
    log_params(eval_params)
    evaluator = RolloutWorker(params["make_env"], policy, **eval_params)
    evaluator.seed(params["rank_seed"])
    return evaluator


def configure_her(params):
    env = cached_make_env(params["make_env"])
    env.reset()

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    her_params = {"reward_fun": reward_fun}
    for name in ["replay_strategy", "replay_k"]:
        her_params[name] = params[name]
        params["_" + name] = her_params[name]
        del params[name]
    logger.info("\n*** her_params ***")
    log_params(her_params)
    sample_her_transitions = make_sample_her_transitions(**her_params)

    return sample_her_transitions


def log_params(params):
    for key in sorted(params.keys()):
        logger.info("{}: {}".format(key, params[key]))


def prepare_params(params):

    env_name = params["env_name"]

    def make_env():
        env = gym.make(env_name)
        return env

    params["make_env"] = make_env
    tmp_env = cached_make_env(params["make_env"])
    assert hasattr(tmp_env, "_max_episode_steps")
    params["T"] = tmp_env._max_episode_steps
    tmp_env.reset()
    params["max_u"] = np.array(params["max_u"]) if isinstance(params["max_u"], list) else params["max_u"]
    params["dims"] = configure_dims(params)
    params["gamma"] = 1.0 - 1.0 / params["T"]
    return params


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b
