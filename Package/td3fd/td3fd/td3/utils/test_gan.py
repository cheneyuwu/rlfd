import os
import random
import numpy as np
import torch

import sklearn.datasets
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from td3fd.td3.shaping import GANShaping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_image(gan, true_dist, idx):
    # generates and saves a plot of the true distribution, the generator, and the critic
    N_POINTS = 128
    RANGE = 2

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype="float32")
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    # if FIXED_GENERATOR is not True:
    #     samples = sess.run(fake_data, feed_dict={real_data: points})
    # disc_map = sess.run(disc_real, feed_dict={real_data: points})
    points = torch.tensor(points).to(device)
    disc_map = gan.D(points)
    disc_map = disc_map.cpu().data.numpy()

    plt.clf()
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.colorbar()  # add color bar

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c="orange", marker="+")
    # if FIXED_GENERATOR is not True:
    #     plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='*')
    plt.show()
    # plt.savefig(os.path.join("temp", str(idx).zfill(3) + '.jpg'))


# Dataset iterator
def inf_train_gen():
    if dataset == "8gaussians":
        scale = 2.0
        centers = [
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]

        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            batch_data = []
            for _ in range(BATCH_SIZE):
                point = np.random.randn(2) * 0.02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                batch_data.append(point)

            batch_data = np.array(batch_data, dtype=np.float32)
            batch_data /= 1.414  # std
            yield batch_data

    elif dataset == "25gaussians":
        batch_data = []
        for i_ in range(4000):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    batch_data.append(point)

        batch_data = np.asarray(batch_data, dtype=np.float32)
        np.random.shuffle(batch_data)
        batch_data /= 2.828  # std

        while True:
            for i_ in range(int(len(batch_data) / BATCH_SIZE)):
                yield batch_data[i_ * BATCH_SIZE : (i_ + 1) * BATCH_SIZE]

    elif dataset == "swissroll":
        while True:
            batch_data = sklearn.datasets.make_swiss_roll(n_samples=BATCH_SIZE, noise=0.25)[0]
            batch_data = batch_data.astype(np.float32)[:, [0, 2]]
            batch_data /= 7.5  # stdev plus a little
            yield batch_data


if __name__ == "__main__":
    dataset = "8gaussians"
    BATCH_SIZE = 64

    data_gen = inf_train_gen()
    gan = GANShaping({"o": (2,), "g": (0,), "u": (0,)}, 1.0, 0.99, 1.0)
    for i in range(10000):
        data = data_gen.__next__()
        batch_data = {"o": data, "g": np.empty((BATCH_SIZE, 0)), "u": np.empty((BATCH_SIZE, 0))}
        if i % 1000 == 0:
            generate_image(gan, data, 0)
        gan.train(batch_data)
