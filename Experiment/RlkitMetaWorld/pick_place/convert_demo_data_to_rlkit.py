import numpy as np
import pickle

# demo data to be converted
data = dict(np.load("demo_data_td3fd.npz"))

print("Demo data shape before conversion:")
for k, v in data.items():
    print(k, v.shape)

# Convert to 2d format first
if len(data["o"].shape) == 3:
    print("Demo data from parallel envs, convert to 2D arrays first")
    interm_data = data
    data = {}
    # observations
    o = interm_data.pop("o")
    data["o"] = o[:, :-1, ...]
    data["o_2"] = o[:, 1:, ...]
    # achieved goals
    ag = interm_data.pop("ag")
    data["ag"] = ag[:, :-1, ...]
    data["ag_2"] = ag[:, 1:, ...]
    # goals
    g = interm_data.pop("g")
    data["g"] = g
    data["g_2"] = g
    # others which are the same
    for k, v in interm_data.items():
        data[k] = v

    # Now we need to convert them to 2D
    for k, v in data.items():
        data[k] = v.reshape(v.shape[0] * v.shape[1], v.shape[2])
    
    # the missed done signal
    data["done"] = np.zeros_like(data["r"])

    print("Demo data shape after first conversion:")
    for k, v in data.items():
        print(k, v.shape)

# converted data
converted_data = {}
converted_data["observations"] = np.concatenate((data["o"], data["g"]), axis=1)
converted_data["next_observations"] = np.concatenate((data["o_2"], data["g_2"]), axis=1)
converted_data["actions"] = data["u"]
converted_data["rewards"] = data["r"]
converted_data["terminals"] = data["done"]
converted_data["agent_infos"] = []
converted_data["env_infos"] = []
for i in range(data["u"].shape[0]):
    env_infos = {k.replace("info_", ""): v[i] for k, v in data.items() if k.startswith("info")}
    agent_infos = {}
    converted_data["agent_infos"].append(agent_infos)
    converted_data["env_infos"].append(env_infos)

# Sanity check
print("Demo data after final conversion:")
print(converted_data.keys())
print("observation shape:", converted_data["observations"].shape)
print(converted_data["env_infos"][0])
print("should be 1.0 of smaller than 1.0:", np.max(converted_data["actions"]))  # should be 1


pickle.dump([converted_data], open("demo_data.npz", "wb"))  # switch to a list
print("Demo file has been stored into demo_data.npz")


print("Loading check")
data = pickle.load(open("demo_data.npz", "rb"))
print(len(data))
