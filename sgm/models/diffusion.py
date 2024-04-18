from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union, DefaultDict

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange
import math
import torch.nn as nn
from ..modules import UNCONDITIONAL_CONFIG
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)


import collections
from functools import partial


def save_activations(
        activations: DefaultDict,
        name: str,
        module: nn.Module,
        inp: Tuple,
        out: torch.Tensor
) -> None:
    """PyTorch Forward hook to save outputs at each forward
    pass. Mutates specified dict objects with each fwd pass.
    """
    if isinstance(out, tuple):
        if out[1] is None:
            activations[name].append(out[0].detach())


class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        trainkeys='pose',
        multiplier=0.05,
        loss_rgb_lambda=20.,
        loss_fg_lambda=10.,
        loss_bg_lambda=20.,
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.trainkeys = trainkeys
        self.multiplier = multiplier
        self.loss_rgb_lambda = loss_rgb_lambda
        self.loss_fg_lambda = loss_fg_lambda
        self.loss_bg_lambda = loss_bg_lambda
        self.rgb = network_config.params.rgb
        self.rgb_predict = network_config.params.rgb_predict
        self.add_token = ('modifier_token' in conditioner_config.params.emb_models[1].params)
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        blocks = []
        if self.trainkeys == 'poseattn':
            for x in self.model.diffusion_model.named_parameters():
                if not ('pose' in x[0] or 'transformer_blocks' in x[0]):
                    x[1].requires_grad = False
                else:
                    if 'pose' in x[0]:
                        x[1].requires_grad = True
                        blocks.append(x[0].split('.pose')[0])

            blocks = set(blocks)
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' in x[0]:
                    reqgrad = False
                    for each in blocks:
                        if each in x[0] and ('attn1' in x[0] or 'attn2' in x[0] or 'pose' in x[0]):
                            reqgrad = True
                            x[1].requires_grad = True
                    if not reqgrad:
                        x[1].requires_grad = False
        elif self.trainkeys == 'pose':
            for x in self.model.diffusion_model.named_parameters():
                if not ('pose' in x[0]):
                    x[1].requires_grad = False
                else:
                    x[1].requires_grad = True
        elif self.trainkeys == 'all':
            for x in self.model.diffusion_model.named_parameters():
                x[1].requires_grad = True

        self.model = self.model.to(memory_format=torch.channels_last)

    def register_activation_hooks(
            self,
    ) -> None:
        self.activations_dict = collections.defaultdict(list)
        handles = []
        for name, module in self.model.diffusion_model.named_modules():
            if len(name.split('.')) > 1 and name.split('.')[-2] == 'transformer_blocks':
                if hasattr(module, 'pose_emb_layers'):
                    handle = module.register_forward_hook(
                        partial(save_activations, self.activations_dict, name)
                    )
                    handles.append(handle)
        self.handles = handles

    def clear_rendered_feat(self,):
        for name, module in self.model.diffusion_model.named_modules():
            if len(name.split('.')) > 1 and name.split('.')[-2] == 'transformer_blocks':
                if hasattr(module, 'pose_emb_layers'):
                    module.rendered_feat = None

    def remove_activation_hooks(
            self, handles
    ) -> None:
        for handle in handles:
            handle.remove()

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        return batch[self.input_key], batch[self.input_key + '_ref'] if self.input_key + '_ref' in batch else None, batch['pose'] if 'pose' in batch else None, batch['mask'] if "mask" in batch else None, batch['mask_ref'] if "mask_ref" in batch else None, batch['depth'] if "depth" in batch else None, batch['drop_im'] if "drop_im" in batch else 0.

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            out = self.first_stage_model.decode(z)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            z = self.first_stage_model.encode(x)
        z = self.scale_factor * z
        return z

    def forward(self, x, x_rgb, xr, pose, mask, mask_ref, opacity, drop_im, batch):
        loss, loss_fg, loss_bg, loss_rgb = self.loss_fn(self.model, self.denoiser, self.conditioner, x, x_rgb, xr, pose, mask, mask_ref, opacity, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean.item()}
        if self.rgb and self.global_step > 0:
            loss_fg = (loss_fg.mean(1)*drop_im.reshape(-1)).sum()/(drop_im.sum() + 1e-12)
            loss_bg = (loss_bg.mean(1)*drop_im.reshape(-1)).sum()/(drop_im.sum() + 1e-12)
            loss_mean += self.loss_fg_lambda*loss_fg
            loss_mean += self.loss_bg_lambda*loss_bg
            loss_dict["loss_fg"] = loss_fg.item()
            loss_dict["loss_bg"] = loss_bg.item()
        if self.rgb_predict and loss_rgb.mean() > 0:
            loss_rgb = (loss_rgb.mean(1)*drop_im.reshape(-1)).sum()/(drop_im.sum() + 1e-12)
            loss_mean += self.loss_rgb_lambda*loss_rgb
            loss_dict["loss_rgb"] = loss_rgb.item()
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x, xr, pose, mask, mask_ref, opacity, drop_im = self.get_input(batch)
        x_rgb = x.clone().detach()
        x = self.encode_first_stage(x)
        x = x.to(memory_format=torch.channels_last)
        if xr is not None:
            b, n = xr.shape[0], xr.shape[1]
            xr = rearrange(self.encode_first_stage(rearrange(xr, "b n ... -> (b n) ...")), "(b n) ... -> b n ...", b=b, n=n)
            xr = drop_im.reshape(b, 1, 1, 1, 1)*xr + (1-drop_im.reshape(b, 1, 1, 1, 1))*torch.zeros_like(xr)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, x_rgb, xr, pose, mask, mask_ref, opacity, drop_im, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )
        return loss

    def validation_step(self, batch, batch_idx):
        # print("validation data", len(self.trainer.val_dataloaders))
        loss, loss_dict = self.shared_step(batch)
        return loss

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        blocks = []
        lowlrparams = []
        if self.trainkeys == 'poseattn':
            lowlrparams = []
            for x in self.model.diffusion_model.named_parameters():
                if ('pose' in x[0]):
                    params += [x[1]]
                    blocks.append(x[0].split('.pose')[0])
                    print(x[0])
            blocks = set(blocks)
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' in x[0]:
                    for each in blocks:
                        if each in x[0] and not ('pose' in x[0]) and ('attn1' in x[0] or 'attn2' in x[0]):
                            lowlrparams += [x[1]]
        elif self.trainkeys == 'pose':
            for x in self.model.diffusion_model.named_parameters():
                if ('pose' in x[0]):
                    params += [x[1]]
                    print(x[0])
        elif self.trainkeys == 'all':
            lowlrparams = []
            for x in self.model.diffusion_model.named_parameters():
                if ('pose' in x[0]):
                    params += [x[1]]
                    print(x[0])
                else:
                    lowlrparams += [x[1]]

        for i, embedder in enumerate(self.conditioner.embedders[:2]):
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
            if self.add_token:
                if i == 0:
                    for name, param in embedder.transformer.get_input_embeddings().named_parameters():
                        param.requires_grad = True
                        print(name, "conditional model param")
                        params += [param]
                else:
                    for name, param in embedder.model.token_embedding.named_parameters():
                        param.requires_grad = True
                        print(name, "conditional model param")
                        params += [param]

        if len(lowlrparams) > 0:
            print("different optimizer groups")
            opt = self.instantiate_optimizer_from_config([{'params': params}, {'params': lowlrparams, 'lr': self.multiplier*lr}], lr, self.optimizer_config)
        else:
            opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        num_steps=None,
        randn=None,
        shape: Union[None, Tuple, List] = None,
        return_rgb=False,
        mask=None,
        init_im=None,
        **kwargs,
    ):
        if randn is None:
            randn = torch.randn(batch_size, *shape)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        if mask is not None:
            samples, rgb_list = self.sampler(denoiser, randn.to(self.device), cond, uc=uc, mask=mask, init_im=init_im, num_steps=num_steps)
        else:
            samples, rgb_list = self.sampler(denoiser, randn.to(self.device), cond, uc=uc, num_steps=num_steps)
        if return_rgb:
            return samples, rgb_list
        return samples

    @torch.no_grad()
    def samplemulti(
        self,
        cond,
        uc=None,
        batch_size: int = 16,
        num_steps=None,
        randn=None,
        shape: Union[None, Tuple, List] = None,
        return_rgb=False,
        mask=None,
        init_im=None,
        multikwargs=None,
    ):
        if randn is None:
            randn = torch.randn(batch_size, *shape)

        samples, rgb_list = self.sampler(self.denoiser, self.model, randn.to(self.device), cond, uc=uc, num_steps=num_steps, multikwargs=multikwargs)
        if return_rgb:
            return samples, rgb_list
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int, refernce: bool = True) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if refernce:
                check = (embedder.input_keys[0] in self.log_keys)
            else:
                check = (embedder.input_key in self.log_keys)
            if (
                (self.log_keys is None) or check
            ) and not self.no_cond_log:
                if refernce:
                    x = batch[embedder.input_keys[0]][:n]
                else:
                    x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                if refernce:
                    log[embedder.input_keys[0]] = xc
                else:
                    log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        log = dict()

        x, xr, pose, mask, mask_ref, depth, drop_im = self.get_input(batch)

        if xr is not None:
            conditioner_input_keys = [e.input_keys for e in self.conditioner.embedders]
        else:
            conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]

        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        zr = None
        if xr is not None:
            xr = xr.to(self.device)[:N]
            b, n = xr.shape[0], xr.shape[1]
            log["reference"] = rearrange(xr, "b n ... -> (b n) ...", b=b, n=n)
            zr = rearrange(self.encode_first_stage(rearrange(xr, "b n ... -> (b n) ...", b=b, n=n)), "(b n) ... -> b n ...", b=b, n=n)

        log["inputs"] = x
        b = x.shape[0]
        if mask is not None:
            log["mask"] = mask
        if depth is not None:
            log["depth"] = depth
        z = self.encode_first_stage(x)

        if uc is not None:
            if xr is not None:
                zr = torch.cat([torch.zeros_like(zr), zr])
            drop_im = torch.cat([drop_im, drop_im])
            if isinstance(pose, list):
                pose = pose[:N]*2
            else:
                pose = torch.cat([pose[:N]] * 2)

        sampling_kwargs = {'input_ref': zr}
        sampling_kwargs['pose'] = pose
        sampling_kwargs['mask_ref'] = None
        sampling_kwargs['drop_im'] = drop_im

        log["reconstructions"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N, refernce=True if xr is not None else False))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                if xr is not None:
                    c[k], uc[k] = map(lambda y: y[k][:(n+1)*N].to(self.device), (c, uc))
                else:
                    c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))
        if sample:
            with self.ema_scope("Plotting"):
                samples, rgb_list = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, return_rgb=True, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
            if len(rgb_list) > 0:
                size = int(math.sqrt(rgb_list[0].size(1)))
                log["predicted_rgb"] = rgb_list[0].reshape(-1, size, size, 3).permute(0, 3, 1, 2)
        return log
