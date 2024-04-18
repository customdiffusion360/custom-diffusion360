from typing import Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn

from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims, instantiate_from_config
from .denoiser import Denoiser


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)

        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.loss_type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")


class StandardDiffusionLossImgRef(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        sigma_sampler_config_ref: dict,
        type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.sigma_sampler_ref = None
        if sigma_sampler_config_ref is not None:
            self.sigma_sampler_ref = instantiate_from_config(sigma_sampler_config_ref)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, input_rgb, input_ref, pose, mask, mask_ref, opacity, batch):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(input.shape[0], device=input.device), input.ndim
            )

        additional_model_inputs['pose'] = pose
        additional_model_inputs['mask_ref'] = mask_ref

        noised_input = input + noise * append_dims(sigmas, input.ndim)
        if self.sigma_sampler_ref is not None:
            sigmas_ref = self.sigma_sampler_ref(input.shape[0]).to(input.device)
            if input_ref is not None:
                noise = torch.randn_like(input_ref)
                if self.offset_noise_level > 0.0:
                    noise = noise + self.offset_noise_level * append_dims(
                        torch.randn(input_ref.shape[0], device=input_ref.device), input_ref.ndim
                    )
                input_ref = input_ref + noise * append_dims(sigmas_ref, input_ref.ndim)
            additional_model_inputs['sigmas_ref'] = sigmas_ref

        additional_model_inputs['input_ref'] = input_ref

        model_output, fg_mask_list, alphas, predicted_rgb_list = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )

        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, fg_mask_list, predicted_rgb_list, input, input_rgb, w, mask, mask_ref, opacity, alphas)

    def get_loss(self, model_output, fg_mask_list, predicted_rgb_list, target, target_rgb, w, mask, mask_ref, opacity, alphas_list):
        loss_rgb = []
        loss_fg = []
        loss_bg = []
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            if self.type == "l2":
                loss = (w * (model_output - target) ** 2)
                if mask is not None:
                    loss_l2 = (loss*mask).sum([1, 2, 3])/(mask.sum([1, 2, 3]) + 1e-6)
                else:
                    loss_l2 = torch.mean(loss.reshape(target.shape[0], -1), 1)
                if len(fg_mask_list) > 0 and len(alphas_list) > 0:
                    for fg_mask, alphas in zip(fg_mask_list, alphas_list):
                        size = int(math.sqrt(fg_mask.size(1)))
                        opacity = torch.nn.functional.interpolate(opacity, size=size, antialias=True, mode='bilinear').detach()
                        fg_mask = torch.clamp(fg_mask.reshape(-1, size*size), 0., 1.)
                        loss_fg_ = ((fg_mask - opacity.reshape(-1, size*size))**2).mean(1) #torch.nn.functional.binary_cross_entropy(rgb, torch.clip(mask.reshape(-1, size*size), 0., 1.), reduce=False)
                        loss_bg_ = (alphas - opacity.reshape(-1, size*size, 1, 1)).abs()*(1-opacity.reshape(-1, size*size, 1, 1)) #alpahs  : b hw d 1
                        loss_bg_ = (loss_bg_*((opacity.reshape(-1, size*size, 1, 1) < 0.1)*1)).mean([1, 2, 3])
                        loss_fg.append(loss_fg_)
                        loss_bg.append(loss_bg_)
                    loss_fg = torch.stack(loss_fg, 1)
                    loss_bg = torch.stack(loss_bg, 1)

                if len(predicted_rgb_list) > 0:
                    for rgb in predicted_rgb_list:
                        size = int(math.sqrt(rgb.size(1)))
                        mask_ = torch.nn.functional.interpolate(mask, size=size, antialias=True, mode='bilinear').detach()
                        loss_rgb_ = ((torch.nn.functional.interpolate(target_rgb*0.5+0.5, size=size, antialias=True, mode='bilinear').detach() - rgb.reshape(-1, size, size, 3).permute(0, 3, 1, 2)) ** 2)
                        loss_rgb.append((loss_rgb_*mask_).sum([1, 2, 3])/(mask.sum([1, 2, 3]) + 1e-6))
                    loss_rgb = torch.stack(loss_rgb, 1)
                # print(loss_l2, loss_fg, loss_bg, loss_rgb)
                return loss_l2,  loss_fg, loss_bg, loss_rgb
            elif self.type == "l1":
                return torch.mean(
                    (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
                ),  loss_rgb
            elif self.type == "lpips":
                loss = self.lpips(model_output, target).reshape(-1)
                return loss,  loss_rgb
