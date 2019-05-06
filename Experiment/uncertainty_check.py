"""Script for demonstration neural net uncertainty check
"""

import os
import numpy as np

import matplotlib

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from train import Experiment, Train


def generate_toy_data(num_points, input_range=(-np.pi, np.pi), func=np.sin, random=False):
    """Generated some toy input data for the demonstration neural net work
    """
    if random:
        x = np.random.uniform(*input_range, num_points).reshape(-1, 1)
    else:
        x = np.linspace(*input_range, num_points).reshape(-1, 1)
    y = func(x).reshape(-1, 1)
    return x, y


if __name__ == "__main__":

    exp = Train()

    # Generate some toy regression data
    def func(x):
        return np.sin(4*x)

    # ["training", "test"]
    train_points = 7
    test_points = 100
    data_points = [train_points, test_points]
    random = [True, False]
    result_dir = os.path.join(exp.result_dir, "uncertainty_check")
    os.makedirs(result_dir, exist_ok=True)
    file_name = [result_dir + "/training_set.npz", result_dir + "/test_set.npz"]
    obs = [
        # np.random.uniform(-np.pi, np.pi, 10).reshape(-1, 1),
        np.array((-np.pi/3 - 0.1, -np.pi/3 + 0.3, -0.2, 0, 0.2, np.pi/3 - 0.3, np.pi/3 + 0.1)).reshape(-1,1),
        # np.linspace(-np.pi / 2, np.pi / 2, train_points).reshape(-1, 1),
        np.linspace(-np.pi/2, np.pi/2, test_points).reshape(-1, 1),
        ]

    for i in range(2):
        o = obs[i]
        q = func(o).reshape(-1, 1)
        if not i:
            train_o = o
        u = np.zeros((data_points[i], 1))
        g = np.zeros((data_points[i], 1))
        data_set = {"o": o, "g": g, "u": u, "q": q}
        np.savez_compressed(file_name[i], **data_set)  # save the file

    assert not exp.demo_critic_only(
        env="FakeData",
        logdir=result_dir,
        save_path=result_dir,
        override_params=1,
        demo_data_size=data_points[0],
        demo_batch_size=data_points[0],
        demo_file=result_dir + "/training_set.npz",
        demo_test_file=result_dir + "/test_set.npz",
        train_demo_epochs=100,
        demo_num_sample=12,
        save_interval=1000,
    )

    # Load random data from npz file.
    res = {**np.load(result_dir + "/output/query_latest.npz")}

    # Plot the result
    fig = plt.figure()
    ax = fig.add_subplot(111)
    test_o = res["o"].reshape(-1)
    q_mean = res["q_mean"].reshape(-1)
    q_std = np.sqrt(res["q_var"]).reshape(-1)
    ax.plot(test_o, q_mean, "r-", label="Mean")
    ax.fill_between(test_o, q_mean - q_std, q_mean + q_std, color="grey")
    ax.plot(train_o, func(train_o), "bo", label="Train")
    ax.plot(test_o, func(test_o), "g-", label="Real")
    ax.legend()
    ax.set_title("Critic vs Demonstration NN Output")
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    plt.show(block=False)
    plt.pause(2)
    save_path = os.path.join(result_dir, "uncertainty_check.png")
    plt.savefig(save_path)
    print("Save image to "+save_path)