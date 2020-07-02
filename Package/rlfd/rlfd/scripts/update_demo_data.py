import numpy as np

demo_data = dict(np.load("demo_data.npz"))
np.savez_compressed("demo_data_old.npz", **demo_data)  # save the file

print("Old shapes")
for k, v in demo_data.items():
  print(k, v.shape)

demo_data["o_2"] = demo_data["o"][:, 1:, ...]
demo_data["o"] = demo_data["o"][:, :-1, ...]
if "ag" in demo_data.keys():
  demo_data["ag_2"] = demo_data["ag"][:, 1:, ...]
  demo_data["ag"] = demo_data["ag"][:, :-1, ...]
if "g" in demo_data.keys():
  demo_data["g_2"] = demo_data["g"][:, :, ...]

print("Updated shapes")
for k, v in demo_data.items():
  print(k, v.shape)

np.savez_compressed("demo_data.npz", **demo_data)  # save the file