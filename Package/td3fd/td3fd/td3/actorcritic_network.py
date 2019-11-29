import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, dimo, dimg, dimu, max_u, layer_sizes):
        super().__init__()

        input_size = dimo[0] + dimg[0]
        output_size = dimu[0]

        layers = []
        layers.append(nn.Linear(input_size, layer_sizes[0]))
        layers.append(nn.ReLU())
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        layers.append(nn.Tanh())
        self.layers = nn.ModuleList(layers)

        self.max_u = max_u

    def forward(self, o, g):
        res = torch.cat([o, g], axis=1)
        for l in self.layers:
            res = l(res)
        return res * self.max_u


class Critic(nn.Module):
    def __init__(self, dimo, dimg, dimu, max_u, layer_sizes):
        super().__init__()

        input_size = dimo[0] + dimg[0] + dimu[0]
        output_size = 1

        layers = []
        layers.append(nn.Linear(input_size, layer_sizes[0]))
        layers.append(nn.ReLU())
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        self.layers = nn.ModuleList(layers)

        self.max_u = max_u

    def forward(self, o, g, u):
        res = torch.cat([o, g, u / self.max_u], axis=1)
        for l in self.layers:
            res = l(res)
        return res


def test_actor():
    dimo, dimg, dimu = 2, 1, 1
    layer_sizes = [16]
    actor = Actor(dimo, dimg, dimu, layer_sizes).to(device)
    summary(actor, [(dimo,), (dimg,)])


def test_critic():
    dimo, dimg, dimu = 2, 1, 1
    layer_sizes = [16]
    critic = Critic(dimo, dimg, dimu, layer_sizes).to(device)
    summary(critic, [(dimo,), (dimg,), (dimu,)])


if __name__ == "__main__":
    test_actor()
    test_critic()
