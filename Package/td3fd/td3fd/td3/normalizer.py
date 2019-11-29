"""adopted from openai baseline code base, converted to pytorch
"""
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Normalizer(torch.nn.Module):
    def __init__(self, shape, eps=0.0, clip_range=np.inf):
        """
        A normalizer that ensures that observations are approximately distributed according to a standard Normal
        distribution (i.e. have mean zero and variance one).

        Args:
            shape      (tuple)  - the shape of the observation to be normalized
            eps        (float)  - a small constant that avoids underflows
            clip_range (float)  - normalized observations are clipped to be in [-clip_range, clip_range]
        """
        super().__init__()

        self.shape = shape
        self.clip_range = clip_range

        self.register_buffer("eps", torch.tensor(eps, requires_grad=False))
        self.register_buffer("sum_tc", torch.zeros(*self.shape, requires_grad=False))
        self.register_buffer("sumsq_tc", torch.zeros(*self.shape, requires_grad=False))
        self.register_buffer("count_tc", torch.zeros(1, requires_grad=False))
        self.register_buffer("mean_tc", torch.zeros(*self.shape, requires_grad=False))
        self.register_buffer("std_tc", torch.ones(*self.shape, requires_grad=False))

    def update(self, v):
        v = v.reshape(-1, *self.shape)
        self.count_tc += v.shape[0]
        self.sum_tc += torch.sum(v, dim=0)
        self.sumsq_tc += torch.sum(v ** 2, dim=0)

        self.mean_tc = self.sum_tc / self.count_tc
        self.std_tc = torch.sqrt(
            torch.max(self.eps ** 2, self.sumsq_tc / self.count_tc - (self.sum_tc / self.count_tc) ** 2)
        )

    def normalize(self, v):
        assert type(v) == torch.Tensor, type(v)
        return torch.clamp((v - self.mean_tc) / self.std_tc, -self.clip_range, self.clip_range)

    def denormalize(self, v):
        assert type(v) == torch.Tensor, type(v)
        return self.mean_tc + v * self.std_tc


def test_normalizer():
    normalizer = Normalizer((1,)).to(device)
    dist_tc = torch.distributions.normal.Normal(loc=torch.tensor(1.0).to(device), scale=torch.tensor(2.0).to(device))
    train_data = dist_tc.sample([1000000])
    test_data = dist_tc.sample([1000000])
    normalizer.update(train_data)
    print(normalizer.mean_tc, normalizer.std_tc)
    output = normalizer.normalize(test_data)
    print(torch.mean(output, dim=0), torch.std(output, dim=0))
    revert_output = normalizer.denormalize(output)
    print(torch.mean(revert_output, dim=0), torch.std(revert_output, dim=0))


if __name__ == "__main__":
    import time

    t = time.time()
    test_normalizer()
    t = time.time() - t
    print("Time: ", t)
