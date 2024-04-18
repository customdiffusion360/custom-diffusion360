import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from ..modules.utils_cameraray import (
    get_patch_rays,
    get_plucker_parameterization,
    positional_encoding,
    convert_to_view_space,
    convert_to_view_space_points,
    convert_to_target_space,
)


from pytorch3d.renderer import ray_bundle_to_ray_points
from pytorch3d.renderer.implicit.raysampling import RayBundle as RayBundle
from pytorch3d import _C

from ..modules.diffusionmodules.util import zero_module


class FeatureNeRFEncoding(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        far_plane: float = 2.0,
        rgb_predict=False,
        average=False,
        num_freqs=16,
    ) -> None:
        super().__init__()

        self.far_plane = far_plane
        self.rgb_predict = rgb_predict
        self.average = average
        self.num_freqs = num_freqs
        dim = 3
        self.plane_coefs = nn.Sequential(
            nn.Linear(in_channels + self.num_freqs * dim * 4 + 2 * dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        if not self.average:
            self.nviews = nn.Linear(
                in_channels + self.num_freqs * dim * 4 + 2 * dim, 1
            )
        self.decoder = zero_module(
                nn.Linear(out_channels, 1 + (3 if rgb_predict else 0), bias=False)
            )

    def forward(self, pose, xref, ray_points, rays, mask_ref):
        # xref : [b, n, hw, c]
        # ray_points: [b, n+1, hw, d, 3]
        # rays: [b, n+1, hw, 6]

        b, n, hw, c = xref.shape
        d = ray_points.shape[3]
        res = int(math.sqrt(hw))
        if mask_ref is not None:
            mask_ref = torch.nn.functional.interpolate(
                rearrange(
                    mask_ref,
                    "b n ... -> (b n) ...",
                ),
                size=[res, res],
                mode="nearest",
            ).reshape(b, n, -1, 1)
            xref = xref * mask_ref

        volume = []
        for i, cam in enumerate(pose):
            volume.append(
                cam.transform_points_ndc(ray_points[i, 0].reshape(-1, 3)).reshape(n + 1, hw, d, 3)[..., :2]
            )
        volume = torch.stack(volume)

        plane_features = F.grid_sample(
            rearrange(
                xref,
                "b n (h w) c -> (b n) c h w",
                b=b,
                h=int(math.sqrt(hw)),
                w=int(math.sqrt(hw)),
                c=c,
                n=n,
            ),
            torch.clip(
                torch.nan_to_num(
                    rearrange(-1 * volume[:, 1:].detach(), "b n ... -> (b n) ...")
                ),
                -1.2,
                1.2,
            ),
            align_corners=True,
            padding_mode="zeros",
        )  # [bn, c, hw, d]

        plane_features = rearrange(plane_features, "(b n) ... -> b n ...", b=b, n=n)

        xyz_grid_features_inviewframe = convert_to_view_space_points(pose, ray_points[:, 0])
        xyz_grid_features_inviewframe_encoding = positional_encoding(xyz_grid_features_inviewframe, self.num_freqs)
        camera_features_inviewframe = (
            convert_to_view_space(pose, rays[:, 0])[:, 1:]
            .reshape(b, n, hw, 1, -1)
            .expand(-1, -1, -1, d, -1)
        )
        camera_features_inviewframe_encoding = positional_encoding(
            get_plucker_parameterization(camera_features_inviewframe),
            self.num_freqs // 2,
        )
        xyz_grid_features = xyz_grid_features_inviewframe_encoding[:, :1].expand(
            -1, n, -1, -1, -1
        )
        camera_features = (
            (convert_to_target_space(pose, rays[:, 1:])[..., :3])
            .reshape(b, n, hw, 1, -1)
            .expand(-1, -1, -1, d, -1)
        )
        camera_features_encoding = positional_encoding(
            camera_features, self.num_freqs
        )
        plane_features_final = self.plane_coefs(
                torch.cat(
                    [
                        plane_features.permute(0, 1, 3, 4, 2),
                        xyz_grid_features_inviewframe_encoding[:, 1:],
                        xyz_grid_features_inviewframe[:, 1:],
                        camera_features_inviewframe_encoding,
                        camera_features_inviewframe[..., 3:],
                    ],
                    dim=-1,
                )
            )  # b, n, hw, d, c

        # plane_features = torch.cat([plane_features, xyz_grid_features, camera_features], dim=1)
        if not self.average:
            plane_features_attn = nn.functional.softmax(
                self.nviews(
                    torch.cat(
                        [
                            plane_features.permute(0, 1, 3, 4, 2),
                            xyz_grid_features,
                            xyz_grid_features_inviewframe[:, :1].expand(-1, n, -1, -1, -1),
                            camera_features,
                            camera_features_encoding,
                        ],
                        dim=-1,
                    )
                ),
                dim=1,
            )  # b, n, hw, d, 1

            plane_features_final = (plane_features_final * plane_features_attn).sum(1)
        else:
            plane_features_final = plane_features_final.mean(1)
            plane_features_attn = None

        out = self.decoder(plane_features_final)
        return torch.cat([plane_features_final, out], dim=-1), plane_features_attn


class VolRender(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def get_weights(self, densities, deltas):
        """Return weights based on predicted densities

        Args:
            densities: Predicted densities for samples along ray

        Returns:
            Weights for each sample
        """
        delta_density = deltas * densities  # [b, hw, "num_samples", 1]
        alphas = 1 - torch.exp(-delta_density)
        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [
                torch.zeros((*transmittance.shape[:2], 1, 1), device=densities.device),
                transmittance,
            ],
            dim=-2,
        )
        transmittance = torch.exp(-transmittance)  # [b, hw, "num_samples", 1]

        weights = alphas * transmittance  # [b, hw, "num_samples", 1]
        weights = torch.nan_to_num(weights)
        # opacities = 1.0 - torch.prod(1.0 - alphas, dim=-2, keepdim=True)
        return weights, alphas, transmittance

    def forward(
        self,
        features,
        densities,
        dists=None,
        return_weight=False,
        densities_uniform=None,
        dists_uniform=None,
        return_weights_uniform=False,
        rgb=None
    ):
        alphas = None
        fg_mask = None
        if dists is not None:
            weights, alphas, transmittance = self.get_weights(densities, dists)
            fg_mask = torch.sum(weights, -2)
        else:
            weights = densities  # used when we have a pretraind nerf with direct weights as output

        rendered_feats = torch.sum(weights * features, dim=-2) + torch.sum(
            (1 - weights) * torch.zeros_like(features), dim=-2
        )
        if rgb is not None:
            rgb = torch.sum(weights * rgb, dim=-2) + torch.sum(
                (1 - weights) * torch.zeros_like(rgb), dim=-2
            )
        # print("RENDER", fg_mask.shape, weights.shape)
        weights_uniform = None
        if return_weight:
            return rendered_feats, fg_mask, alphas, weights, rgb
        elif return_weights_uniform:
            if densities_uniform is not None:
                weights_uniform, _, transmittance = self.get_weights(densities_uniform, dists_uniform)
            return rendered_feats, fg_mask, alphas, weights_uniform, rgb
        else:
            return rendered_feats, fg_mask, alphas, None, rgb


class Raymarcher(nn.Module):
    def __init__(
        self,
        num_samples=32,
        far_plane=2.0,
        stratified=False,
        training=True,
        imp_sampling_percent=0.9,
        near_plane=0.,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.far_plane = far_plane
        self.near_plane = near_plane
        u_max = 1. / (self.num_samples)
        u = torch.linspace(0, 1 - u_max, self.num_samples, device="cuda")
        self.register_buffer("u", u)
        lengths = torch.linspace(self.near_plane, self.near_plane+self.far_plane, self.num_samples+1, device="cuda")
        # u = (u[..., :-1] + u[..., 1:]) / 2.0
        lengths_center = (lengths[..., 1:] + lengths[..., :-1]) / 2.0
        lengths_upper = torch.cat([lengths_center, lengths[..., -1:]], -1)
        lengths_lower = torch.cat([lengths[..., :1], lengths_center], -1)
        self.register_buffer("lengths", lengths)
        self.register_buffer("lengths_center", lengths_center)
        self.register_buffer("lengths_upper", lengths_upper)
        self.register_buffer("lengths_lower", lengths_lower)
        self.stratified = stratified
        self.training = training
        self.imp_sampling_percent = imp_sampling_percent

    @torch.no_grad()
    def importance_sampling(self, cdf, num_rays, num_samples, device):
        # sample target rays for each reference view
        cdf = cdf[..., 0] + 0.01
        if cdf.shape[1] != num_rays:
            size = int(math.sqrt(num_rays))
            size_ = int(math.sqrt(cdf.size(1)))
            cdf = rearrange(
                torch.nn.functional.interpolate(
                    rearrange(
                        cdf.permute(0, 2, 1), "... (h w) -> ... h w", h=size_, w=size_
                    ),
                    size=[size, size],
                    antialias=True,
                    mode="bilinear",
                ),
                "... h w -> ... (h w)",
                h=size,
                w=size,
            ).permute(0, 2, 1)

        lengths = self.lengths[None, None, :].expand(cdf.shape[0], num_rays, -1)

        cdf_sum = torch.sum(cdf, dim=-1, keepdim=True)
        padding = torch.relu(1e-5 - cdf_sum)
        cdf = cdf + padding / cdf.shape[-1]
        cdf_sum += padding

        pdf = cdf / cdf_sum

        # sample_pdf function
        u_max = 1. / (num_samples)
        u = self.u[None, None, :].expand(cdf.shape[0], num_rays, -1)
        if self.stratified and self.training:
            u += torch.rand((cdf.shape[0], num_rays, num_samples), dtype=cdf.dtype, device=cdf.device,) * u_max

        _C.sample_pdf(
                lengths.reshape(-1, num_samples + 1),
                pdf.reshape(-1, num_samples),
                u.reshape(-1, num_samples),
                1e-5,
            )
        return u, torch.cat([u[..., 1:] - u[..., :-1], lengths[..., -1:] - u[..., -1:]], -1)

    @torch.no_grad()
    def stratified_sampling(self, num_rays, device, uniform=False):
        lengths_uniform = self.lengths[None, None, :].expand(-1, num_rays, -1)

        if uniform:
            return (
                    (lengths_uniform[..., 1:] + lengths_uniform[..., :-1]) / 2.0,
                    lengths_uniform[..., 1:] - lengths_uniform[..., :-1],
                )
        if self.stratified and self.training:
            t_rand = torch.rand(
                (num_rays, self.num_samples + 1),
                dtype=lengths_uniform.dtype,
                device=lengths_uniform.device,
            )
            jittered = self.lengths_lower[None, None, :].expand(-1, num_rays, -1) + \
                (self.lengths_upper[None, None, :].expand(-1, num_rays, -1) - self.lengths_lower[None, None, :].expand(-1, num_rays, -1)) * t_rand
            return ((jittered[..., :-1] + jittered[..., 1:])/2., jittered[..., 1:] - jittered[..., :-1])
        else:
            return (
                    (lengths_uniform[..., 1:] + lengths_uniform[..., :-1]) / 2.0,
                    lengths_uniform[..., 1:] - lengths_uniform[..., :-1],
                )

    @torch.no_grad()
    def forward(self, pose, resolution, weights, imp_sample_next_step=False, device='cuda', pytorch3d=True):
        input_patch_rays, xys = get_patch_rays(
            pose,
            num_patches_x=resolution,
            num_patches_y=resolution,
            device=device,
            return_xys=True,
            stratified=self.stratified and self.training,
        )  # (b, n, h*w, 6)

        num_rays = resolution**2
        # sample target rays for each reference view
        if weights is not None:
            if self.imp_sampling_percent <= 0:
                lengths, dists = self.stratified_sampling(num_rays, device)
            elif (torch.rand(1) < (1.-self.imp_sampling_percent)) and self.training:
                lengths, dists = self.stratified_sampling(num_rays, device)
            else:
                lengths, dists = self.importance_sampling(
                    weights, num_rays, self.num_samples, device=device
                )
        else:
            lengths, dists = self.stratified_sampling(num_rays, device)

        dists_uniform = None
        ray_points_uniform = None
        if imp_sample_next_step:
            lengths_uniform, dists_uniform = self.stratified_sampling(
                num_rays, device, uniform=True
            )

            target_patch_raybundle_uniform = RayBundle(
                origins=input_patch_rays[:, :1, :, :3],
                directions=input_patch_rays[:, :1, :, 3:],
                lengths=lengths_uniform,
                xys=xys.to(device),
            )
            ray_points_uniform = ray_bundle_to_ray_points(target_patch_raybundle_uniform).detach()
            dists_uniform = dists_uniform.detach()

        # print(
        #     "SAMPLING",
        #     lengths.shape,
        #     lengths_uniform.shape,
        #     dists.shape,
        #     dists_uniform.shape,
        #     input_patch_rays.shape,
        # )
        target_patch_raybundle = RayBundle(
            origins=input_patch_rays[:, :1, :, :3],
            directions=input_patch_rays[:, :1, :, 3:],
            lengths=lengths.to(device),
            xys=xys.to(device),
        )
        ray_points = ray_bundle_to_ray_points(target_patch_raybundle)
        return (
            input_patch_rays.detach(),
            ray_points.detach(),
            dists.detach(),
            ray_points_uniform,
            dists_uniform,
        )


class NerfSDModule(nn.Module):
    def __init__(
        self,
        mode="feature-nerf",
        out_channels=None,
        far_plane=2.0,
        num_samples=32,
        rgb_predict=False,
        average=False,
        num_freqs=16,
        stratified=False,
        imp_sampling_percent=0.9,
        near_plane=0.
    ):
        MODES = {
            "feature-nerf": FeatureNeRFEncoding,  # ampere
        }
        super().__init__()
        self.rgb_predict = rgb_predict

        self.raymarcher = Raymarcher(
            num_samples=num_samples,
            far_plane=near_plane + far_plane,
            stratified=stratified,
            imp_sampling_percent=imp_sampling_percent,
            near_plane=near_plane,
        )
        model_class = MODES[mode]
        self.model = model_class(
            out_channels,
            out_channels,
            far_plane=near_plane + far_plane,
            rgb_predict=rgb_predict,
            average=average,
            num_freqs=num_freqs,
        )

    def forward(self, pose, xref=None, mask_ref=None, prev_weights=None, imp_sample_next_step=False,):
        # xref: b n h w c or b n hw c
        # pose: a list of pytorch3d cameras
        # mask_ref: mask corresponding to black regions because of padding non square images.
        rgb = None
        dists_uniform = None
        weights_uniform = None
        resolution = (int(math.sqrt(xref.size(2))) if len(xref.shape) == 4 else xref.size(3))
        input_patch_rays, ray_points, dists, ray_points_uniform, dists_uniform = (self.raymarcher(pose, resolution, weights=prev_weights, device=xref.device))
        output, plane_features_attn = self.model(pose, xref, ray_points, input_patch_rays, mask_ref)
        weights = output[..., -1:]
        features = output[..., :-1]
        if self.rgb_predict:
            rgb = features[..., -3:]
            features = features[..., :-3]
        dists = dists.unsqueeze(-1)
        with torch.no_grad():
            if ray_points_uniform is not None:
                output_uniform, _ = self.model(pose, xref, ray_points_uniform, input_patch_rays, mask_ref)
                weights_uniform = output_uniform[..., -1:]
                dists_uniform = dists_uniform.unsqueeze(-1)

        return (
            features,
            weights,
            dists,
            plane_features_attn,
            rgb,
            weights_uniform,
            dists_uniform,
        )
