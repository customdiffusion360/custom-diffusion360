import argparse
import copy
import glob
import os
import sys
from typing import List

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch_lightning import seed_everything

sys.path.append('./')
from sgm.modules.utils_cameraray import (
    interpolate_translate_interpolate_xaxis,
    interpolate_translate_interpolate_yaxis,
    interpolate_translate_interpolate_zaxis,
    interpolatefocal,
)
from sgm.util import load_model_from_config

choices = []


def get_unique_embedder_keys_from_conditioner(conditioner):
    p = [x.input_keys for x in conditioner.embedders]
    return list(set([item for sublist in p for item in sublist])) + ['jpg_ref']


def customforward(self, x, xr, context=None, contextr=None, pose=None, mask_ref=None, prev_weights=None, timesteps=None):
    if not isinstance(context, list):
        context = [context]
    b, c, h, w = x.shape
    x_in = x
    fg_masks = []
    alphas = []
    rgbs = []

    x = self.norm(x)

    if not self.use_linear:
        x = self.proj_in(x)

    x = rearrange(x, "b c h w -> b (h w) c").contiguous()
    if self.use_linear:
        x = self.proj_in(x)

    prev_weights = None
    counter = 0
    for i, block in enumerate(self.transformer_blocks):
        if i > 0 and len(context) == 1:
            i = 0  # use same context for each block
        if self.image_cross and (counter % self.poscontrol_interval == 0):
            x, fg_mask, weights, alpha, rgb = block(x, context=context[i], context_ref=x, pose=pose, mask_ref=mask_ref, prev_weights=prev_weights)
            prev_weights = weights
            fg_masks.append(fg_mask)
            if alpha is not None:
                alphas.append(alpha)
            if rgb is not None:
                rgbs.append(rgb)
        else:
            x, _, _, _, _ = block(x, context=context[i])
        counter += 1
    if self.use_linear:
        x = self.proj_out(x)
    x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
    if not self.use_linear:
        x = self.proj_out(x)
    if len(fg_masks) > 0:
        if len(rgbs) <= 0:
            rgbs = None
        if len(alphas) <= 0:
            alphas = None
        return x + x_in, None, fg_masks, prev_weights, alphas, rgbs
    else:
        return x + x_in, None, None, prev_weights, None, None


def _customforward(
        self, x, context=None, context_ref=None, pose=None, mask_ref=None, prev_weights=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
     ):
    if context_ref is not None:
        global choices
        batch_size = x.size(0)
        # IP2P like sampling or default sampling
        if batch_size % 3 == 0:
            batch_size = batch_size // 3
            context_ref = torch.stack([self.references[:-1][y] for y in choices]).unsqueeze(0).expand(batch_size, -1, -1, -1)
            context_ref = torch.cat([self.references[-1:].unsqueeze(0).expand(batch_size, context_ref.size(1), -1, -1), context_ref, context_ref], dim=0)
        else:
            batch_size = batch_size // 2
            context_ref = torch.stack([self.references[:-1][y] for y in choices]).unsqueeze(0).expand(batch_size, -1, -1, -1)
            context_ref = torch.cat([self.references[-1:].unsqueeze(0).expand(batch_size, context_ref.size(1), -1, -1), context_ref], dim=0)

    fg_mask = None
    weights = None
    alphas = None
    predicted_rgb = None

    x = (
        self.attn1(
            self.norm1(x),
            context=context if self.disable_self_attn else None,
            additional_tokens=additional_tokens,
            n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
            if not self.disable_self_attn
            else 0,
        )
        + x
    )

    x = (
            self.attn2(
                self.norm2(x), context=context, additional_tokens=additional_tokens,
            )
            + x
        )

    if context_ref is not None:
        if self.rendered_feat is not None:
            x = self.pose_emb_layers(torch.cat([x, self.rendered_feat], dim=-1))
        else:
            xref, fg_mask, weights, alphas, predicted_rgb = self.reference_attn(x,
                                                                                context_ref,
                                                                                context,
                                                                                pose,
                                                                                prev_weights,
                                                                                mask_ref)
            self.rendered_feat = xref
            x = self.pose_emb_layers(torch.cat([x, xref], -1))

    x = self.ff(self.norm3(x)) + x
    return x, fg_mask, weights, alphas, predicted_rgb


def log_images(
        model,
        batch,
        N: int = 1,
        noise=None,
        scale_im=3.5,
        num_steps: int = 10,
        ucg_keys: List[str] = None,
        **kwargs,
        ):

    log = dict()
    conditioner_input_keys = [e.input_keys for e in model.conditioner.embedders]
    ucg_keys = conditioner_input_keys
    pose = batch['pose']

    c, uc = model.conditioner.get_unconditional_conditioning(
        batch,
        force_uc_zero_embeddings=ucg_keys
        if len(model.conditioner.embedders) > 0
        else [],
        force_ref_zero_embeddings=True
    )

    _, n = 1, len(pose)-1
    sampling_kwargs = {}

    if scale_im > 0:
        if uc is not None:
            if isinstance(pose, list):
                pose = pose[:N]*3
            else:
                pose = torch.cat([pose[:N]] * 3)
    else:
        if uc is not None:
            if isinstance(pose, list):
                pose = pose[:N]*2
            else:
                pose = torch.cat([pose[:N]] * 2)

    sampling_kwargs['pose'] = pose
    sampling_kwargs['drop_im'] = None
    sampling_kwargs['mask_ref'] = None

    for k in c:
        if isinstance(c[k], torch.Tensor):
            c[k], uc[k] = map(lambda y: y[k][:(n+1)*N].to('cuda'), (c, uc))

    import time
    st = time.time()
    with model.ema_scope("Plotting"):
        samples = model.sample(
                c, shape=noise.shape[1:], uc=uc, batch_size=N, num_steps=num_steps, noise=noise,  **sampling_kwargs
            )
        model.clear_rendered_feat()
    samples = model.decode_first_stage(samples)
    print("Time taken for sampling", time.time() - st)
    log["samples"] = samples.cpu()

    return log


def sample(config,
           ckpt=None,
           delta_ckpt=None,
           camera_path=None,
           num_images=6,
           prompt_list=None,
           scale=7.5,
           num_ref=8,
           num_steps=50,
           output_dir='',
           scale_im=None,
           max_images=20,
           seed=30,
           specific_id='',
           interp_start=-0.2,
           interp_end=0.21,
           interp_step=0.4,
           translateY=False,
           translateZ=False,
           translateX=False,
           translate_focal=False,
           resolution=512,
           random_render_path=None,
           allround_render=False,
           equidistant=False,
           ):

    config = OmegaConf.load(config)

    # setup guider config
    if scale_im > 0:
        guider_config = {'target': 'sgm.modules.diffusionmodules.guiders.ScheduledCFGImgTextRef',
                         'params': {'scale': scale, 'scale_im': scale_im}
                         }
        config.model.params.sampler_config.params.guider_config = guider_config
    else:
        guider_config = {'target': 'sgm.modules.diffusionmodules.guiders.VanillaCFGImgRef',
                         'params': {'scale': scale}
                         }
        config.model.params.sampler_config.params.guider_config = guider_config

    # load model
    model = load_model_from_config(config, ckpt, delta_ckpt)
    model = model.cuda()

    # change forward methods to store rendered features from first step and use the pre-calculated reference features
    def register_recr(net_):
        if net_.__class__.__name__ == 'SpatialTransformer':
            bound_method = customforward.__get__(net_, net_.__class__)
            setattr(net_, 'forward', bound_method)
            return
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                register_recr(net__)
        return

    def register_recr2(net_):
        if net_.__class__.__name__ == 'BasicTransformerBlock':
            bound_method = _customforward.__get__(net_, net_.__class__)
            setattr(net_, 'forward', bound_method)
            return
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                register_recr2(net__)
        return

    sub_nets = model.model.diffusion_model.named_children()
    for net in sub_nets:
        register_recr(net[1])
        register_recr2(net[1])

    # load cameras
    cameras_val, cameras_train = torch.load(camera_path)
    global choices
    num_ref = 8
    max_diff = len(cameras_train)/num_ref
    choices = [int(x) for x in torch.linspace(0, len(cameras_train) - max_diff, num_ref)]
    cameras_train_final = [cameras_train[i] for i in choices]

    # start sampling
    model.clear_rendered_feat()
    seedeval_counter = seed
    counter = 0
    for _, prompt in enumerate(prompt_list):
        curent_seed = seedeval_counter
        seed_everything(curent_seed)

        if translateZ or translateY or translateX or translate_focal:
            interp_reps = len(np.arange(interp_start, interp_end, interp_step))
            noise = torch.randn(1, 4, resolution // 8, resolution // 8).to('cuda').repeat(num_images*interp_reps, 1, 1, 1)
        else:
            noise = torch.randn(1, 4, resolution // 8, resolution // 8).to('cuda').repeat(num_images, 1, 1, 1)

        # random sample camera poses
        pose_ids = np.random.choice(len(cameras_val), num_images, replace=False)
        print(pose_ids)
        pose = [cameras_val[i] for i in pose_ids]

        # prepare batches [if translating then call required functions on the target pose]
        batches = []
        for i in range(num_images):
            batch = {'pose': [pose[i]] + cameras_train_final,
                     "original_size_as_tuple": torch.tensor([512, 512]).reshape(-1, 2),
                     "target_size_as_tuple": torch.tensor([512, 512]).reshape(-1, 2),
                     "crop_coords_top_left": torch.tensor([0, 0]).reshape(-1, 2),
                     "original_size_as_tuple_ref": torch.tensor([512, 512]).reshape(-1, 2),
                     "target_size_as_tuple_ref": torch.tensor([512, 512]).reshape(-1, 2),
                     "crop_coords_top_left_ref": torch.tensor([0, 0]).reshape(-1, 2),
                     }
            if translateZ or translateY or translateX or translate_focal:
                cameras = []
                if translateY:
                    cameras += interpolate_translate_interpolate_yaxis(batch["pose"][0], interp_start, interp_end, interp_step)
                elif translateZ:
                    cameras += interpolate_translate_interpolate_zaxis(batch["pose"][0], interp_start, interp_end, interp_step)
                elif translateX:
                    cameras += interpolate_translate_interpolate_xaxis(batch["pose"][0], interp_start, interp_end, interp_step)
                else:
                    cameras += interpolatefocal(batch["pose"][0], interp_start, interp_end, interp_step)
                for j in range(len(cameras)):
                    batch_ = copy.deepcopy(batch)
                    batch_["pose"][0] = cameras[j]
                    batch_["pose"] = [join_cameras_as_batch(batch_["pose"])]
                    batches.append(batch_)
            else:
                batch["pose"] = [join_cameras_as_batch(batch["pose"])]
                batches.append(batch)

        print(f'len batches: {len(batches)}')
        with torch.no_grad():
            for batch in batches:
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to('cuda')
                    elif 'pose' in key:
                        batch[key] = [x.to('cuda') for x in batch[key]]
                    else:
                        pass

                batch["txt"] = [prompt for _ in range(1)]
                batch["txt_ref"] = [prompt for _ in range(len(batch["pose"])-1)]
                print(batch["txt"])

                N = 1
                log_ = log_images(model, batch, N=N, noise=noise.clone()[:N], num_steps=50, scale_im=scale_im)
                im = Image.fromarray((torch.clip(log_["samples"][0].permute(1, 2, 0)*0.5+0.5, 0, 1.).cpu().numpy()*255).astype(np.uint8))
                prompt_ = prompt.replace(' ', '_')
                im.save(f'{output_dir}/sample_{counter}_{prompt_}_{seedeval_counter}.png')
                counter += 1
                torch.cuda.empty_cache()
    return


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default='pretrained-models/sd_xl_base_1.0.safetensors')
    parser.add_argument("--custom_model_dir", type=str, default=None)
    parser.add_argument("--translateY", action="store_true")
    parser.add_argument("--translateZ", action="store_true")
    parser.add_argument("--translateX", action="store_true")
    parser.add_argument("--translate_focal", action="store_true")
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--num_ref", type=int, default=8)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--scale_im", type=float, default=3.5)
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--interp_start", type=float, default=-0.2)
    parser.add_argument("--interp_end", type=float, default=0.21)
    parser.add_argument("--interp_step", type=float, default=0.4)
    parser.add_argument("--allround_render", action="store_true")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)

    args.delta_ckpt = os.path.join(args.custom_model_dir, 'checkpoints', 'step=000001600.ckpt')
    args.config = sorted(glob.glob(os.path.join(args.custom_model_dir, "configs/*.yaml")))[-1]
    args.camera_path = os.path.join(args.custom_model_dir, 'camera.bin')
    sample(args.config,
           ckpt=args.ckpt,
           delta_ckpt=args.delta_ckpt,
           camera_path=args.camera_path,
           num_images=args.num_images,
           prompt_list=[args.prompt],
           scale=args.scale,
           num_ref=args.num_ref,
           num_steps=args.num_steps,
           output_dir=args.output_dir,
           scale_im=args.scale_im,
           seed=args.seed,
           interp_start=args.interp_start,
           interp_end=args.interp_end,
           interp_step=args.interp_step,
           translateY=args.translateY,
           translateZ=args.translateZ,
           translateX=args.translateX,
           translate_focal=args.translate_focal,
           allround_render=args.allround_render,
           )
