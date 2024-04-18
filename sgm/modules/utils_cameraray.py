#### Code taken from: https://github.com/mayankgrwl97/gbt
"""Utils for ray manipulation"""

import numpy as np
import torch
from pytorch3d.renderer.implicit.raysampling import RayBundle as RayBundle
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import PerspectiveCameras


############################# RAY BUNDLE UTILITIES #############################

def is_scalar(x):
    """Returns True if the provided variable is a scalar

    Args:
        x: scalar or array-like (numpy array or torch tensor)

    Returns:
        bool: True if x is of the type scalar, or array-like with 0 dimension. False, otherwise

    """
    if isinstance(x, float) or isinstance(x, int):
        return True

    if isinstance(x, np.ndarray) and np.ndim(x) == 0:
        return True

    if isinstance(x, torch.Tensor) and x.dim() == 0:
        return True

    return False


def transform_rays(reference_R, reference_T, rays):
    """
    PyTorch3D Convention is used: X_cam = X_world @ R + T

    Args:
        reference_R: world2cam rotation matrix for reference camera (B, 3, 3)
        reference_T: world2cam translation vector for reference camera (B, 3)
        rays: (origin, direction) defined in world reference frame (B, V, N, 6)
    Returns:
        torch.Tensor: Transformed rays w.r.t. reference camera (B, V, N, 6)
    """
    batch, num_views, num_rays, ray_dim = rays.shape
    assert (
        ray_dim == 6
    ), "First 3 dimensions should be origin; Last 3 dimensions should be direction"

    rays = rays.reshape(batch, num_views * num_rays, ray_dim)
    rays_out = rays.clone()
    rays_out[..., :3] = torch.bmm(rays[..., :3], reference_R) + reference_T.unsqueeze(
        -2
    )
    rays_out[..., 3:] = torch.bmm(rays[..., 3:], reference_R)
    rays_out = rays_out.reshape(batch, num_views, num_rays, ray_dim)
    return rays_out


def get_directional_raybundle(cameras, x_pos_ndc, y_pos_ndc, max_depth=1):
    if is_scalar(x_pos_ndc):
        x_pos_ndc = [x_pos_ndc]
    if is_scalar(y_pos_ndc):
        y_pos_ndc = [y_pos_ndc]
    assert is_scalar(max_depth)

    if not isinstance(x_pos_ndc, torch.Tensor):
        x_pos_ndc = torch.tensor(x_pos_ndc)  # (N, )
    if not isinstance(y_pos_ndc, torch.Tensor):
        y_pos_ndc = torch.tensor(y_pos_ndc)  # (N, )

    xy_depth = torch.stack(
        (x_pos_ndc, y_pos_ndc, torch.ones_like(x_pos_ndc) * max_depth), dim=-1
    )  # (N, 3)

    num_points = xy_depth.shape[0]

    unprojected = cameras.unproject_points(
        xy_depth.to(cameras.device), world_coordinates=True, from_ndc=True
    )  # (N, 3)
    unprojected = unprojected.unsqueeze(0).to("cpu")  # (B, N, 3)

    origins = (
        cameras.get_camera_center()[:, None, :].expand(-1, num_points, -1).to("cpu")
    )  # (B, N, 3)
    directions = unprojected - origins  # (B, N, 3)
    directions = directions / directions.norm(dim=-1).unsqueeze(-1)  # (B, N, 3)
    lengths = (
        torch.tensor([[0, 3]]).unsqueeze(0).expand(-1, num_points, -1).to("cpu")
    )  # (B, N, 2)
    xys = xy_depth[:, :2].unsqueeze(0).to("cpu")  # (B, N, 2)

    raybundle = RayBundle(
        origins=origins.to("cpu"),
        directions=directions.to("cpu"),
        lengths=lengths.to("cpu"),
        xys=xys.to("cpu"),
    )
    return raybundle


def get_patch_raybundle(
    cameras, num_patches_x, num_patches_y, max_depth=1, stratified=False
):
    horizontal_patch_edges = torch.linspace(1, -1, num_patches_x + 1)
    # horizontal_positions = horizontal_patch_edges[:-1]  # (num_patches_x,): Top left corner of patch

    vertical_patch_edges = torch.linspace(1, -1, num_patches_y + 1)
    # vertical_positions = vertical_patch_edges[:-1]  # (num_patches_y,): Top left corner of patch
    if stratified:
        horizontal_patch_edges_center = (
            horizontal_patch_edges[..., 1:] + horizontal_patch_edges[..., :-1]
        ) / 2.0
        horizontal_patch_edges_upper = torch.cat(
            [horizontal_patch_edges_center, horizontal_patch_edges[..., -1:]], -1
        )
        horizontal_patch_edges_lower = torch.cat(
            [horizontal_patch_edges[..., :1], horizontal_patch_edges_center], -1
        )
        horizontal_positions = (
            horizontal_patch_edges_lower
            + (horizontal_patch_edges_upper - horizontal_patch_edges_lower)
            * torch.rand_like(horizontal_patch_edges_lower)
        )[..., :-1]

        vertical_patch_edges_center = (
            vertical_patch_edges[..., 1:] + vertical_patch_edges[..., :-1]
        ) / 2.0
        vertical_patch_edges_upper = torch.cat(
            [vertical_patch_edges_center, vertical_patch_edges[..., -1:]], -1
        )
        vertical_patch_edges_lower = torch.cat(
            [vertical_patch_edges[..., :1], vertical_patch_edges_center], -1
        )
        vertical_positions = (
            vertical_patch_edges_lower
            + (vertical_patch_edges_upper - vertical_patch_edges_lower)
            * torch.rand_like(vertical_patch_edges_lower)
        )[..., :-1]
    else:
        horizontal_positions = (
            horizontal_patch_edges[:-1] + horizontal_patch_edges[1:]
        ) / 2  # (num_patches_x, )  # Center of patch
        vertical_positions = (
            vertical_patch_edges[:-1] + vertical_patch_edges[1:]
        ) / 2  # (num_patches_y, )  # Center of patch

    h_pos, v_pos = torch.meshgrid(
        horizontal_positions, vertical_positions, indexing='xy'
    )  # (num_patches_y, num_patches_x), (num_patches_y, num_patches_x)
    h_pos = h_pos.reshape(-1)  # (num_patches_y * num_patches_x)
    v_pos = v_pos.reshape(-1)  # (num_patches_y * num_patches_x)

    raybundle = get_directional_raybundle(
        cameras=cameras, x_pos_ndc=h_pos, y_pos_ndc=v_pos, max_depth=max_depth
    )
    return raybundle


def get_patch_rays(
    cameras_list,
    num_patches_x,
    num_patches_y,
    device,
    return_xys=False,
    stratified=False,
):
    """Returns patch rays given the camera viewpoints

    Args:
        cameras_list(list[pytorch3d.renderer.cameras.BaseCameras]): List of list of cameras (len (batch_size, num_input_views,))
        num_patches_x: Number of patches in the x-direction (horizontal)
        num_patches_y: Number of patches in the y-direction (vertical)

    Returns:
        torch.tensor: Patch rays of shape (batch_size, num_views, num_patches, 6)
    """
    batch, numviews = len(cameras_list), len(cameras_list[0])
    cameras_list = join_cameras_as_batch([cam for cam_batch in cameras_list for cam in cam_batch])
    patch_rays = get_patch_raybundle(
                    cameras_list,
                    num_patches_y=num_patches_y,
                    num_patches_x=num_patches_x,
                    stratified=stratified,
                )
    if return_xys:
        xys = patch_rays.xys

    patch_rays = torch.cat((patch_rays.origins.unsqueeze(0), patch_rays.directions), dim=-1)
    patch_rays = patch_rays.reshape(
        batch, numviews, num_patches_x * num_patches_y, 6
    ).to(device)
    if return_xys:
        return patch_rays, xys
    return patch_rays

############################ RAY PARAMETERIZATION ##############################


def get_plucker_parameterization(ray):
    """Returns the plucker representation of the rays given the (origin, direction) representation

    Args:
        ray(torch.Tensor): Tensor of shape (..., 6) with the (origin, direction) representation

    Returns:
        torch.Tensor: Tensor of shape (..., 6) with the plucker (D, OxD) representation
    """
    ray = ray.clone()  # Create a clone
    ray_origins = ray[..., :3]
    ray_directions = ray[..., 3:]
    ray_directions = ray_directions / ray_directions.norm(dim=-1).unsqueeze(
        -1
    )  # Normalize ray directions to unit vectors
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    plucker_parameterization = torch.cat([ray_directions, plucker_normal], dim=-1)

    return plucker_parameterization


def positional_encoding(ray, n_freqs=10, start_freq=0):
    """
    Positional Embeddings. For more details see Section 5.1 of
    NeRFs: https://arxiv.org/pdf/2003.08934.pdf

    Args:
        ray: (B,P,d)
        n_freqs: num of frequency bands
        parameterize(str|None): Parameterization used for rays. Recommended: use 'plucker'. Default=None.

    Returns:
        pos_embeddings: Mapping input ray from R to R^{2*n_freqs}.
    """
    start_freq = -1 * (n_freqs / 2)
    freq_bands = 2.0 ** torch.arange(start_freq, start_freq + n_freqs) * np.pi
    sin_encodings = [torch.sin(ray * freq) for freq in freq_bands]
    cos_encodings = [torch.cos(ray * freq) for freq in freq_bands]
    pos_embeddings = torch.cat(
        sin_encodings + cos_encodings, dim=-1
    )  # B, P, d * 2n_freqs
    return pos_embeddings


def convert_to_target_space(input_cameras, input_rays):
    input_rays_transformed = []
    # input_cameras: b, N
    # input_rays: b, N, hw, 6
    # return: b, N, hw, 6
    for i in range(len(input_cameras[0])):
        reference_cameras = [cameras[0] for cameras in input_cameras]
        reference_R = [
            camera.R.to(input_rays.device) for camera in reference_cameras
        ]  # List (length=batch_size) of Rs(shape: 1, 3, 3)
        reference_R = torch.cat(reference_R, dim=0)  # (B, 3, 3)
        reference_T = [
            camera.T.to(input_rays.device) for camera in reference_cameras
        ]  # List (length=batch_size) of Ts(shape: 1, 3)
        reference_T = torch.cat(reference_T, dim=0)  # (B, 3)
        input_rays_transformed.append(
            transform_rays(
                reference_R=reference_R,
                reference_T=reference_T,
                rays=input_rays[:, i: i + 1],
            )
        )
    return torch.cat(input_rays_transformed, 1)


def convert_to_view_space(input_cameras, input_rays):
    input_rays_transformed = []
    # input_cameras: b, N
    # input_rays: b, hw, 6
    # return: b, n, hw, 6
    for i in range(len(input_cameras[0])):
        reference_cameras = [cameras[i] for cameras in input_cameras]
        reference_R = [
            camera.R.to(input_rays.device) for camera in reference_cameras
        ]  # List (length=batch_size) of Rs(shape: 1, 3, 3)
        reference_R = torch.cat(reference_R, dim=0)  # (B, 3, 3)
        reference_T = [
            camera.T.to(input_rays.device) for camera in reference_cameras
        ]  # List (length=batch_size) of Ts(shape: 1, 3)
        reference_T = torch.cat(reference_T, dim=0)  # (B, 3)
        input_rays_transformed.append(
            transform_rays(
                reference_R=reference_R,
                reference_T=reference_T,
                rays=input_rays.unsqueeze(1),
            )
        )
    return torch.cat(input_rays_transformed, 1)


def convert_to_view_space_points(input_cameras, input_points):
    input_rays_transformed = []
    # input_cameras: b, N
    # ipput_points: b, hw, d, 3
    # returns: b, N, hw, d, 3 [target points transformed in the reference view frame]
    for i in range(len(input_cameras[0])):
        reference_cameras = [cameras[i] for cameras in input_cameras]
        reference_R = [
            camera.R.to(input_points.device) for camera in reference_cameras
        ]  # List (length=batch_size) of Rs(shape: 1, 3, 3)
        reference_R = torch.cat(reference_R, dim=0)  # (B, 3, 3)
        reference_T = [
            camera.T.to(input_points.device) for camera in reference_cameras
        ]  # List (length=batch_size) of Ts(shape: 1, 3)
        reference_T = torch.cat(reference_T, dim=0)  # (B, 3)
        input_points_clone = torch.einsum(
            "bsdj,bjk->bsdk", input_points, reference_R
        ) + reference_T.reshape(-1, 1, 1, 3)
        input_rays_transformed.append(input_points_clone.unsqueeze(1))
    return torch.cat(input_rays_transformed, dim=1)


def interpolate_translate_interpolate_xaxis(cam1, interp_start, interp_end, interp_step):
    cameras = []
    for i in np.arange(interp_start, interp_end, interp_step):
        viewtoworld = cam1.get_world_to_view_transform().inverse()

        x_axis = torch.from_numpy(np.array([i, 0., 0.0])).reshape(1, 3).float().to(cam1.device)
        newc = viewtoworld.transform_points(x_axis)
        rt = cam1.R[0]
        # t = cam1.T
        new_t = -rt.T@newc.T

        cameras.append(PerspectiveCameras(R=cam1.R,
                                          T=new_t.T,
                                          focal_length=cam1.focal_length,
                                          principal_point=cam1.principal_point,
                                          image_size=512,
                                          )
                       )
    return cameras


def interpolate_translate_interpolate_yaxis(cam1, interp_start, interp_end, interp_step):
    cameras = []
    for i in np.arange(interp_start, interp_end, interp_step):
        # i = np.clip(i, -0.2, 0.18)
        viewtoworld = cam1.get_world_to_view_transform().inverse()

        x_axis = torch.from_numpy(np.array([0, i, 0.0])).reshape(1, 3).float().to(cam1.device)
        newc = viewtoworld.transform_points(x_axis)
        rt = cam1.R[0]
        # t = cam1.T
        new_t = -rt.T@newc.T

        cameras.append(PerspectiveCameras(R=cam1.R,
                                          T=new_t.T,
                                          focal_length=cam1.focal_length,
                                          principal_point=cam1.principal_point,
                                          image_size=512,
                                          )
                       )
    return cameras


def interpolate_translate_interpolate_zaxis(cam1, interp_start, interp_end, interp_step):
    cameras = []
    for i in np.arange(interp_start, interp_end, interp_step):
        viewtoworld = cam1.get_world_to_view_transform().inverse()

        x_axis = torch.from_numpy(np.array([0, 0., i])).reshape(1, 3).float().to(cam1.device)
        newc = viewtoworld.transform_points(x_axis)
        rt = cam1.R[0]
        # t = cam1.T
        new_t = -rt.T@newc.T

        cameras.append(PerspectiveCameras(R=cam1.R,
                                          T=new_t.T,
                                          focal_length=cam1.focal_length,
                                          principal_point=cam1.principal_point,
                                          image_size=512,
                                          )
                       )
    return cameras


def interpolatefocal(cam1, interp_start, interp_end, interp_step):
    cameras = []
    for i in np.arange(interp_start, interp_end, interp_step):
        cameras.append(PerspectiveCameras(R=cam1.R,
                                          T=cam1.T,
                                          focal_length=cam1.focal_length*i,
                                          principal_point=cam1.principal_point,
                                          image_size=512,
                                          )
                       )
    return cameras
