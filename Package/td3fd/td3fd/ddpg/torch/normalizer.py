"""adopted from openai baseline code base, converted to pytorch
"""
import numpy as np
import torch


class Normalizer(torch.nn.Module):
    def __init__(self, size, eps=1e-2, clip_range=np.inf):
        """
        A normalizer that ensures that observations are approximately distributed according to a standard Normal
        distribution (i.e. have mean zero and variance one).

        Args:
            size       (int)    - the size of the observation to be normalized
            eps        (float)  - a small constant that avoids underflows
            clip_range (float)  - normalized observations are clipped to be in [-clip_range, clip_range]
        """
        super().__init__()

        self.size = size
        self.clip_range = clip_range
        self.dtype = torch.float32

        self.register_buffer("eps", torch.tensor(eps, dtype=self.dtype, requires_grad=False))
        self.register_buffer("sum_tc", torch.zeros(self.size, dtype=self.dtype, requires_grad=False))
        self.register_buffer("sumsq_tc", torch.zeros(self.size, dtype=self.dtype, requires_grad=False))
        self.register_buffer("count_tc", torch.zeros(1, dtype=self.dtype, requires_grad=False))
        self.register_buffer("mean_tc", torch.zeros(self.size, dtype=self.dtype, requires_grad=False))
        self.register_buffer("std_tc", torch.ones(self.size, dtype=self.dtype, requires_grad=False))

    def update(self, v):
        v = v.reshape(-1, self.size)
        self.count_tc += v.shape[0]
        self.sum_tc += torch.sum(v, dim=0)
        self.sumsq_tc += torch.sum(torch.mul(v, v), dim=0)

        self.mean_tc = self.sum_tc / self.count_tc
        self.std_tc = torch.sqrt(
            torch.max(
                torch.mul(self.eps, self.eps),
                self.sumsq_tc / self.count_tc - torch.mul(self.sum_tc / self.count_tc, self.sum_tc / self.count_tc),
            )
        )

    def normalize(self, v):
        assert type(v) == torch.Tensor, type(v)
        mean_tc, std_tc = self._reshape_for_broadcasting(v)
        return torch.clamp((v - mean_tc) / std_tc, -self.clip_range, self.clip_range)

    def denormalize(self, v):
        assert type(v) == torch.Tensor, type(v)
        mean_tc, std_tc = self._reshape_for_broadcasting(v)
        return mean_tc + v * std_tc

    def _reshape_for_broadcasting(self, v):
        dim = len(v.shape) - 1
        mean_tc = torch.reshape(self.mean_tc, [1] * dim + [self.size])
        std_tc = torch.reshape(self.std_tc, [1] * dim + [self.size])
        return mean_tc, std_tc


def test_normalizer():
    normalizer = Normalizer(1)
    dist_tc = torch.distributions.normal.Normal(loc=1.0, scale=2.0)
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
