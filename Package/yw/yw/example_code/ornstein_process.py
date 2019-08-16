"""Adopted from https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
"""

import numpy as np
import matplotlib.pyplot as plt


sigma = 1.  # Standard deviation.
mu = 0.  # Mean.
tau = .5  # Time constant.

dt = .001  # Time step.
T = 1.  # Total time.
n = int(T / dt)  # Number of time steps.
t = np.linspace(0., T, n)  # Vector of times.


sigma_bis = sigma * np.sqrt(2. / tau)
sqrtdt = np.sqrt(dt)

x = np.zeros(n)

for i in range(n - 1):
    x[i + 1] = x[i] + dt * (-(x[i] - mu) / tau) + sigma_bis * sqrtdt * np.random.randn()

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, x, lw=2)

plt.show()