"""MPI Utilities
"""

import os
import sys
import subprocess

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import numpy as np
import random


def install_mpi_excepthook():
    import sys
    from mpi4py import MPI

    old_hook = sys.excepthook

    def new_hook(a, b, c):
        old_hook(a, b, c)
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort()

    sys.excepthook = new_hook


def mpi_input(msg, comm=None):
    if MPI is None:
        return input(msg)
    if comm is None:
        comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        ret = input(msg)
    else:
        ret = None
    ret = comm.bcast(ret, root=0)
    return ret


def mpi_exit(code=0, comm=None):
    """Allow 1 process to terminate other processes
    """
    if MPI is None:
        exit(code)
    if comm is None:
        comm = MPI.COMM_WORLD
    comm.Abort(code)


def mpi_average(value, comm=None):
    if MPI is None:
        return value
    if comm is None:
        comm = MPI.COMM_WORLD
    if not isinstance(value, list):
        value = [value]
    elif len(value) == 0:
        value = [0.0]
    return mpi_moments(np.array(value), comm=comm)[0]


def mpi_mean(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    if comm is None:
        comm = MPI.COMM_WORLD
    xsum = x.sum(axis=axis, keepdims=keepdims)
    n = xsum.size
    localsum = np.zeros(n + 1, x.dtype)
    localsum[:n] = xsum.ravel()
    localsum[n] = x.shape[axis]
    globalsum = np.zeros_like(localsum)
    comm.Allreduce(localsum, globalsum, op=MPI.SUM)
    return globalsum[:n].reshape(xsum.shape) / globalsum[n], globalsum[n]


def mpi_moments(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    mean, count = mpi_mean(x, axis=axis, comm=comm, keepdims=True)
    sqdiffs = np.square(x - mean)
    meansqdiff, count1 = mpi_mean(sqdiffs, axis=axis, comm=comm, keepdims=True)
    assert count1 == count
    std = np.sqrt(meansqdiff)
    if not keepdims:
        newshape = mean.shape[:axis] + mean.shape[axis + 1 :]
        mean = mean.reshape(newshape)
        std = std.reshape(newshape)
    return mean, std, count
