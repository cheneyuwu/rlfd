import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)