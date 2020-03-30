import numpy as np
import pickle

# data = dict(np.load("demo_data_td3fd.npz"))

# # Convert to 2d format first
# # episode = {k: v.reshape((-1, v.shape[-1])) for k, v in episode.items()}
# # episode["o_2"] = episode["o"][1:, ...]
# # episode["o"] = episode["o"][:-1, ...]
# # episode["ag_2"] = episode["ag"][1:, ...]
# # episode["ag"] = episode["ag"][:-1, ...]
# # episode["g_2"] = episode["g"][...]
# # done = np.zeros(T)
# # done[-1] = 1.0
# # episode["done"] = done.reshape(-1, 1)

# # for k, v in data.items():
# #     try:
# #         print(k, v.shape)
# #     except:
# #         print(k, v[0])
# # exit()

# # converted data
# converted_data = {}
# converted_data["observations"] = np.concatenate((data["o"], data["g"]), axis=1)
# converted_data["next_observations"] = np.concatenate((data["o_2"], data["g_2"]), axis=1)
# converted_data["actions"] = data["u"] / 2 # TODO
# converted_data["rewards"] = data["r"]
# converted_data["terminals"] = data["done"]
# converted_data["agent_infos"] = []
# converted_data["env_infos"] = []
# for i in range(data["u"].shape[0]):
#     env_infos = {k.replace("info_", ""): v[i] for k, v in data.items() if k.startswith("info")}
#     agent_infos = {}
#     converted_data["agent_infos"].append(agent_infos)
#     converted_data["env_infos"].append(env_infos)

# # array(batch_size x (T or T+1) x dim_key), we only need the first one!
# pickle.dump(paths, open("demo_data.npz", "wb" ))
# print("Demo file has been stored into {}.".format(file_name))

# print(converted_data.keys())
# print(converted_data["observations"].shape)
# print(converted_data["env_infos"][0])
# print(np.max(converted_data["actions"]))

data = pickle.load(open("demo_data.npz", "rb"))
print(len(data))