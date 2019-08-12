import numpy as np

from yw.env.env_manager import EnvManager


def fetch_pick_and_place_demo(env, run_id, render):

    episodeAcs = []
    episodeObs = []
    episodeInfo = []
    episodeReward = []

    last_obs = env.reset()
    goal = last_obs["desired_goal"]
    objectPos = last_obs["observation"][3:6]
    object_rel_pos = last_obs["observation"][6:9]

    timeStep = 0  # count the total number of timesteps

    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03  # first make the gripper go slightly above the object

    episodeObs.append(last_obs)

    # move to the above of the object
    while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
        if render:
            env.render()
        action = [0, 0, 0, 0]
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i] * 10
        action[-1] = 0.05  # open

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        episodeReward.append([reward])

        objectPos = obsDataNew["observation"][3:6]
        object_rel_pos = obsDataNew["observation"][6:9]

    # take the object
    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps:
        if render:
            env.render()
        action = [0, 0, 0, 0]
        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i] * 10

        action[-1] = -0.05

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        episodeReward.append([reward])

        objectPos = obsDataNew["observation"][3:6]
        object_rel_pos = obsDataNew["observation"][6:9]

    # set up the first goal
    # method 1 completly random
    # weight = np.random.uniform(0, 1, size=3)
    # method 2 fixed with little disturbance
    if run_id % 2 == 0:
        weight = np.array((0.8, 0.2, 0.5)) + np.random.normal(scale=0.2, size=3)
    elif run_id % 2 == 1:
        weight = np.array((0.2, 0.8, 0.5)) + np.random.normal(scale=0.2, size=3)
    else:
        assert False
    intermidiate_goal = objectPos * weight + goal * (1 - weight)

    while np.linalg.norm(intermidiate_goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps:
        if render:
            env.render()
        action = [0, 0, 0, 0]
        for i in range(len(intermidiate_goal - objectPos)):
            action[i] = (intermidiate_goal - objectPos)[i] * 10
        action[len(action) - 1] = -0.05

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        episodeReward.append([reward])

        objectPos = obsDataNew["observation"][3:6]
        object_rel_pos = obsDataNew["observation"][6:9]

    # going to the final goal
    while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps:
        if render:
            env.render()
        action = [0, 0, 0, 0]
        for i in range(len(goal - objectPos)):
            action[i] = (goal - objectPos)[i] * 10
        action[len(action) - 1] = -0.05

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        episodeReward.append([reward])

        objectPos = obsDataNew["observation"][3:6]
        object_rel_pos = obsDataNew["observation"][6:9]

    while timeStep <= env._max_episode_steps:  # limit the number of timesteps in the episode to a fixed duration
        if render:
            env.render()
        action = [0, 0, 0, 0]
        action[len(action) - 1] = -0.05  # keep the gripper closed

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        episodeReward.append([reward])

        objectPos = obsDataNew["observation"][3:6]
        object_rel_pos = obsDataNew["observation"][6:9]

    assert episodeInfo[-1]["is_success"]

    return episodeObs, episodeAcs, episodeReward, episodeInfo


def main():

    num_itr = 30
    render = True
    env = EnvManager(env_name="FetchPickAndPlace-v1", env_args={}, r_scale=1.0, r_shift=0.0, eps_length=50).get_env()

    demo_data_obs = []
    demo_data_acs = []
    demo_data_rewards = []
    demo_data_info = []

    for i in range(num_itr):
        print("Iteration number: ", i)
        episodeObs, episodeAcs, episodeReward, episodeInfo = fetch_pick_and_place_demo(env, i, render=render)
        demo_data_obs.append(episodeObs)
        demo_data_acs.append(episodeAcs)
        demo_data_rewards.append(episodeReward)
        demo_data_info.append(episodeInfo)

    T = env._max_episode_steps

    result = None

    for epsd in range(num_itr):  # we initialize the whole demo buffer at the start of the training
        obs, acts, goals, achieved_goals, rs = [], [], [], [], []
        info_keys = [key.replace("info_", "") for key in demo_data_info[0][0].keys()]
        info_values = [np.empty((T, 1, 1), np.float32) for key in info_keys]
        for transition in range(T):
            obs.append([demo_data_obs[epsd][transition].get("observation")])
            acts.append([demo_data_acs[epsd][transition]])
            goals.append([demo_data_obs[epsd][transition].get("desired_goal")])
            achieved_goals.append([demo_data_obs[epsd][transition].get("achieved_goal")])
            rs.append([demo_data_rewards[epsd][transition]])
            for idx, key in enumerate(info_keys):
                info_values[idx][transition, 0] = demo_data_info[epsd][transition][key]

        obs.append([demo_data_obs[epsd][T].get("observation")])
        achieved_goals.append([demo_data_obs[epsd][T].get("achieved_goal")])

        episode = dict(o=obs, u=acts, g=goals, ag=achieved_goals, r=rs)
        for key, value in zip(info_keys, info_values):
            episode["info_{}".format(key)] = value

        # switch to batch major
        episode_batch = {}
        for key in episode.keys():
            val = np.array(episode[key]).copy()
            # make inputs batch-major instead of time-major
            episode_batch[key] = val.swapaxes(0, 1)
        episode = episode_batch

        if result == None:
            result = episode
        else:
            for k in result.keys():
                result[k] = np.concatenate((result[k], episode[k]), axis=0)

    # array(batch_size x (T or T+1) x dim_key), we only need the first one!
    np.savez_compressed("demo_data.npz", **result)  # save the file


if __name__ == "__main__":
    main()

