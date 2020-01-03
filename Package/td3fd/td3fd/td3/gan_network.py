import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, layer_sizes):
        super().__init__()

        layers = []
        layers.append(nn.Linear(latent_dim, layer_sizes[0]))
        layers.append(nn.ReLU())
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, layer_sizes[0]))
        layers.append(nn.ReLU())
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x