import numpy as np
a = dict(np.load("demo_data.npz.bak"))
print(a.keys())
print(a["g"].shape)
print(a["ag"].shape)
a["g"] = np.empty((50, 40, 0))
a["ag"] = np.empty((50, 41, 0))

np.savez_compressed("demo_data.npz", **a)  # save the file