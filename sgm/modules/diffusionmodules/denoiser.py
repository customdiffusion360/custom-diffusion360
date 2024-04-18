import torch.nn as nn
import torch
from ...util import append_dims, instantiate_from_config


class Denoiser(nn.Module):
    def __init__(self, weighting_config, scaling_config):
        super().__init__()

        self.weighting = instantiate_from_config(weighting_config)
        self.scaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma):
        return sigma

    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    def w(self, sigma):
        return self.weighting(sigma)

    def __call__(self, network, input, sigma, cond, sigmas_ref=None, **kwargs):
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        if sigmas_ref is not None:
            if kwargs is not None:
                kwargs['sigmas_ref'] = sigmas_ref
            else:
                kwargs = {'sigmas_ref': sigmas_ref}

            if kwargs['input_ref'] is not None:
                noise = torch.randn_like(kwargs['input_ref'])
                kwargs['input_ref'] = kwargs['input_ref'] + noise * append_dims(sigmas_ref, kwargs['input_ref'].ndim)

        if 'input_ref' in kwargs and kwargs['input_ref'] is not None and 'sigmas_ref' in kwargs:
            _, _, c_in_ref, c_noise_ref = self.scaling(append_dims(kwargs['sigmas_ref'], kwargs['input_ref'].ndim))
            kwargs['input_ref'] = kwargs['input_ref']*c_in_ref
            kwargs['sigmas_ref'] = self.possibly_quantize_c_noise(kwargs['sigmas_ref'])

        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        predict, fg_mask_list, alphas_list, rgb_list = network(input * c_in, c_noise, cond, **kwargs)
        return predict * c_out + input * c_skip, fg_mask_list, alphas_list, rgb_list


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        weighting_config,
        scaling_config,
        num_idx,
        discretization_config,
        do_append_zero=False,
        quantize_c_noise=True,
        flip=True,
    ):
        super().__init__(weighting_config, scaling_config)
        sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
