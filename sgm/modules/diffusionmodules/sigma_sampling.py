import torch

from ...util import default, instantiate_from_config


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples, rand=None):
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        return log_sigma.exp()


class DiscreteSampling:
    def __init__(self, discretization_config, num_idx, num_idx_start=0, do_append_zero=False, flip=True):
        self.num_idx = num_idx
        self.num_idx_start = num_idx_start
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        idx = default(
            rand,
            torch.randint(self.num_idx_start, self.num_idx, (n_samples,)),
        )
        return self.idx_to_sigma(idx)


class CubicSampling:
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True):
        self.num_idx = num_idx
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        t = torch.rand((n_samples,))
        t = (1 - t ** 3) * (self.num_idx-1)
        t = t.long()
        idx = default(
            rand,
            t,
        )
        return self.idx_to_sigma(idx)
