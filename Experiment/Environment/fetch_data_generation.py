import gym
import numpy as np


"""Data generation for the case of a single block pick and place in Fetch Env"""

actions = []
observations = []
infos = []
rewards = []


def main():
    env = gym.make("FetchPickAndPlace-v1")
    numItr = 1000
    env.reset()
    print("Reset!")
    while len(actions) < numItr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)

    T = env._max_episode_steps

    info_keys = [key.replace("info_", "") for key in infos[0][0].keys()]
    info_values = [np.empty((T, 1, 1), np.float32) for key in info_keys]

    demo_data_obs = observations
    demo_data_acs = actions
    demo_data_rewards = rewards
    demo_data_info = infos


    result = None

    for epsd in range(numItr):  # we initialize the whole demo buffer at the start of the training
        obs, acts, goals, achieved_goals, rs = [], [], [], [], []
        i = 0
        for transition in range(T):
            obs.append([demo_data_obs[epsd][transition].get("observation")])
            acts.append([demo_data_acs[epsd][transition]])
            goals.append([demo_data_obs[epsd][transition].get("desired_goal")])
            achieved_goals.append([demo_data_obs[epsd][transition].get("achieved_goal")])
            rs.append([demo_data_rewards[epsd][transition]])
            for idx, key in enumerate(info_keys):
                info_values[idx][transition, i] = demo_data_info[epsd][transition][key]

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

    fileName = "FetchPickAndPlace-v1"
    # fileName += "_" + str(numItr)
    fileName += ".npz"

    np.savez_compressed(fileName, **result)  # save the file


def goToGoal(env, lastObs):

    goal = lastObs["desired_goal"]
    objectPos = lastObs["observation"][3:6]
    object_rel_pos = lastObs["observation"][6:9]
    episodeAcs = []
    episodeObs = []
    episodeInfo = []
    episodeReward = []

    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03  # first make the gripper go slightly above the object

    timeStep = 0  # count the total number of timesteps
    episodeObs.append(lastObs)

    while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
        # env.render()
        action = [0, 0, 0, 0]
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i] * 6 + np.random.normal(scale=0.05)

        action[len(action) - 1] = 0.05  # open

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        episodeReward.append([reward])

        objectPos = obsDataNew["observation"][3:6]
        object_rel_pos = obsDataNew["observation"][6:9]

    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps:
        # env.render()
        action = [0, 0, 0, 0]
        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i] * 6 + np.random.normal(scale=0.05)

        action[len(action) - 1] = -0.01

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        episodeReward.append([reward])

        objectPos = obsDataNew["observation"][3:6]
        object_rel_pos = obsDataNew["observation"][6:9]

    while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps:
        # env.render()
        action = [0, 0, 0, 0]
        for i in range(len(goal - objectPos)):
            action[i] = (goal - objectPos)[i] * 6 + np.random.normal(scale=0.05)

        action[len(action) - 1] = -0.01

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        episodeReward.append([reward])

        objectPos = obsDataNew["observation"][3:6]
        object_rel_pos = obsDataNew["observation"][6:9]

    while True:  # limit the number of timesteps in the episode to a fixed duration
        # env.render()
        action = [0, 0, 0, 0]
        action[len(action) - 1] = -0.01  # keep the gripper closed

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        episodeReward.append([reward])

        objectPos = obsDataNew["observation"][3:6]
        object_rel_pos = obsDataNew["observation"][6:9]

        if timeStep >= env._max_episode_steps:
            break

    if episodeInfo[-1]["is_success"] == 1.0:
        actions.append(episodeAcs)
        observations.append(episodeObs)
        infos.append(episodeInfo)
        rewards.append(episodeReward)
        return True
    return False


if __name__ == "__main__":
    main()
