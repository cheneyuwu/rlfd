import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, dimo, dimg, dimu, max_u, layer_sizes):
        super().__init__()

        # cnn encoder dims: 3x32x32 => flatten(1x4x4)=16
        self.encoder = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (3x32x32)
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (1024x4x4)
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
            nn.Tanh(),
        )

        # same as TD3
        input_size = 16 + dimg[0]
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
        o = self.encoder(o)
        res = torch.cat([o, g], axis=1)
        for l in self.layers:
            res = l(res)
        return res * self.max_u


class Critic(nn.Module):
    def __init__(self, dimo, dimg, dimu, max_u, layer_sizes):
        super().__init__()

        # cnn encoder dims: 3x32x32 => flatten(1x4x4)=16
        self.encoder = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (3x32x32)
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (1024x4x4)
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
            nn.Tanh(),
        )

        # fully connected layers, same as the original TD3 paper
        # input_size = dimo[0] + dimg[0] + dimu[0]
        input_size = 16 + dimg[0] + dimu[0]
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
        o = self.encoder(o)
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
