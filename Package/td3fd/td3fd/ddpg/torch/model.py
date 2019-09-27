import numpy as np
import torch

from td3fd.util.torch_util import check_module


class Actor(torch.nn.Module):
    def __init__(self, dimo, dimg, dimu, layer_sizes, noise):
        super().__init__()

        self.dimg = dimg

        input_size = dimo + dimg
        output_size = dimu

        layers = []
        layers.append(torch.nn.Linear(input_size, layer_sizes[0]))
        layers.append(torch.nn.ReLU())
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(layer_sizes[-1], output_size))
        layers.append(torch.nn.Tanh())
        self.layers = torch.nn.ModuleList(layers)

        self.noise = torch.distributions.Normal(loc=0.0, scale=0.1 if noise else 0.0)

    def forward(self, o, g):
        state = o
        # for multigoal environments, we have goal as another states
        if self.dimg != 0:
            state = torch.cat([state, g], axis=1)
        else:
            assert g is None

        res = state
        for l in self.layers:
            res = l(res)

        res = res + self.noise.sample(res.shape)
        res = torch.clamp(res, -1.0, 1.0)

        return res


class Critic(torch.nn.Module):
    def __init__(self, dimo, dimg, dimu, layer_sizes):
        super().__init__()

        self.dimg = dimg

        input_size = dimo + dimg + dimu
        output_size = 1

        layers = []
        layers.append(torch.nn.Linear(input_size, layer_sizes[0]))
        layers.append(torch.nn.ReLU())
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(layer_sizes[-1], output_size))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, o, g, u):
        state = o
        # for multigoal environments, we have goal as another states
        if self.dimg != 0:
            state = torch.cat([state, g], axis=1)
        else:
            assert g is None

        res = torch.cat([state, u], axis=1)
        for l in self.layers:
            res = l(res)

        return res


def test_actor():
    dimo, dimg, dimu = 2, 1, 1
    layer_sizes = [16]
    noise = True

    actor = Actor(dimo, dimg, dimu, layer_sizes, noise)
    check_module(actor)


def test_critic():
    dimo, dimg, dimu = 2, 1, 1
    layer_sizes = [16]

    critic = Critic(dimo, dimg, dimu, layer_sizes)
    check_module(critic)


if __name__ == "__main__":
    test_actor()
    test_critic()
