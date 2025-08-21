#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from typing import List, Dict, Tuple, Any
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, RenderedImageAndBackwardValues, rasterize_forward_impl
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim_with_center_grad, ssim


def build_rasterization_settings(
        viewpoint_camera,
        pc: GaussianModel,
        pipe,
        bg_color: torch.Tensor,
        scaling_modifier: float = 1.0,
        prepare_for_gsgn_backward: bool = False) -> GaussianRasterizationSettings:

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    return GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=viewpoint_camera.cx,
        cy=viewpoint_camera.cy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        prepare_for_gsgn_backward=prepare_for_gsgn_backward,
        debug=pipe.debug
    )


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    raster_settings = build_rasterization_settings(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        bg_color=bg_color,
        scaling_modifier=scaling_modifier
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, n_contrib_vol_rend, is_gaussian_hit = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "n_contrib_vol_rend": n_contrib_vol_rend,
        "is_gaussian_hit": is_gaussian_hit,
    }


@torch.no_grad()
def render_and_add_to_residual_norm(
    gaussians: GaussianModel,
    viewpoint_stack,
    opt,
    pipe,
    background,
    scaling_modifier: float = 1.0,
    compute_huber_weights: bool = False,
    huber_c: float = 0.4,
    compute_ssim_residuals: bool = False,
    ssim_residual_scale_factor: float = 0.2
):
    bg = torch.rand(3, device="cuda") if opt.random_background else background
    norm = 0

    # loop over all cameras / images
    for cam in viewpoint_stack:
        gt_image = cam.original_image.cuda()

        raster_settings = build_rasterization_settings(
            viewpoint_camera=cam,
            pc=gaussians,
            pipe=pipe,
            bg_color=bg,
            scaling_modifier=scaling_modifier,
            prepare_for_gsgn_backward=False
        )

        # render image
        num_rendered, n_contrib_vol_rend, is_gaussian_hit, color, radii, geomBuffer, binningBuffer, imgBuffer = rasterize_forward_impl(
            means3D=gaussians.get_xyz,
            sh=gaussians.get_features,
            colors_precomp=torch.Tensor([]),
            opacities=gaussians.get_opacity,
            scales=gaussians.get_scaling,
            rotations=gaussians.get_rotation,
            cov3Ds_precomp=torch.Tensor([]),
            raster_settings=raster_settings
        )

        # calc per-residual error
        residuals = color - gt_image
        if compute_huber_weights:
            # L1 Loss w/o weighted least squares
            # norm += residuals.abs().sum()  # dont have to sqrt() and ** 2 again, since residuals_huber is strictly positive

            # L2 Loss w/o weighted least squares
            # norm += (residuals**2).sum()

            # Huber Loss w/o weighted least squares
            res_abs = residuals.abs()
            mask_outside = res_abs > huber_c
            residuals_huber = 0.5 * res_abs * res_abs
            residuals_huber[mask_outside] = huber_c * (res_abs[mask_outside] - 0.5 * huber_c)

            norm += residuals_huber.sum()  # dont have to sqrt() and ** 2 again, since residuals_huber is strictly positive
        else:
            norm += (residuals ** 2).sum()

        # calculate SSIM residuals / weights
        if compute_ssim_residuals:
            residuals_ssim = ssim(color[None], gt_image[None], size_average=False)
            residuals_ssim = residuals_ssim[0]
            residuals_ssim = 1 - residuals_ssim.clamp_max(1.0)
            norm += ssim_residual_scale_factor * residuals_ssim.sum()  # dont have to sqrt() and ** 2 again, since residuals_huber is strictly positive

    return 0.5 * norm


@torch.enable_grad()
def render_all_images_and_backward(
    gaussians: GaussianModel,
    viewpoint_stack,
    opt,
    pipe,
    background,
    scaling_modifier: float = 1.0,
    prepare_for_gsgn_backward: bool = True,
    compute_huber_weights: bool = False,
    huber_c: float = 0.4,
    compute_ssim_weights: bool = False,
    ssim_residual_scale_factor: float = 0.2
):
    device = gaussians.get_xyz.device
    bg = torch.rand(3, device=device) if opt.random_background else background
    num_images = len(viewpoint_stack)
    out = RenderedImageAndBackwardValues(num_images=num_images)

    # loop over all cameras / images
    residuals_list = []
    weights_list = []
    residuals_ssim_list = []
    weights_ssim_list = []
    for cam in viewpoint_stack:
        # Set up rasterization configuration
        raster_settings = build_rasterization_settings(
            viewpoint_camera=cam,
            pc=gaussians,
            pipe=pipe,
            bg_color=bg,
            scaling_modifier=scaling_modifier,
            prepare_for_gsgn_backward=prepare_for_gsgn_backward
        )

        # render image
        num_rendered, n_contrib_vol_rend, is_gaussian_hit, color, radii, geomBuffer, binningBuffer, imgBuffer = rasterize_forward_impl(
            means3D=gaussians.get_xyz,
            rotations=gaussians.get_rotation,
            scales=gaussians.get_scaling,
            opacities=gaussians.get_opacity,
            sh=gaussians.get_features,
            colors_precomp=torch.Tensor([]),
            cov3Ds_precomp=torch.Tensor([]),
            raster_settings=raster_settings
        )

        # calculate L2 or Huber loss
        gt_image = cam.original_image
        residuals = color - gt_image

        if compute_huber_weights:
            # L1 Loss
            # r = (residuals.abs()).sqrt()
            # dL_dpix = torch.sign(residuals) / (2 * r)
            # dL_dpix[r == 0] = 0

            # L2 Loss
            # r = (residuals**2).sqrt()
            # dL_dpix = torch.sign(residuals)

            # Huber Loss
            res_abs = residuals.abs()
            mask_outside = res_abs > huber_c
            residuals_huber = 0.5 * res_abs * res_abs
            residuals_huber[mask_outside] = huber_c * (res_abs[mask_outside] - 0.5 * huber_c)
            dL_dhuber = residuals
            dL_dhuber[mask_outside] = torch.sign(residuals[mask_outside]) * huber_c

            r = residuals_huber.sqrt()
            dL_dpix = (1.0 / (2 * r)) * dL_dhuber
            dL_dpix[r == 0] = 0

            residuals_list.append(r)
            weights_list.append(dL_dpix)
        else:
            residuals_list.append(residuals)

        # calculate SSIM residuals / weights
        if compute_ssim_weights:
            residuals_ssim, ssim_center_grads = ssim_with_center_grad(color[None], gt_image[None], size_average=False)
            residuals_ssim = residuals_ssim[0]
            ssim_center_grads = -1.0 * ssim_center_grads[0]

            mask_ssim = residuals_ssim > 1.0
            residuals_ssim[mask_ssim] = 1.0
            ssim_center_grads[mask_ssim] = 0
            residuals_ssim = 1 - residuals_ssim

            r = (ssim_residual_scale_factor * residuals_ssim).sqrt()
            dL_dpix = (1.0 / (2 * r)) * (ssim_residual_scale_factor * ssim_center_grads)
            dL_dpix[r == 0] = 0

            residuals_ssim_list.append(r)
            weights_ssim_list.append(dL_dpix)

        # check shared data is equal across all raster_settings
        if out.bg is None:
            out.bg = raster_settings.bg.contiguous()
        else:
            assert torch.allclose(out.bg, raster_settings.bg)

        if out.scale_modifier < 0:
            out.scale_modifier = raster_settings.scale_modifier
        else:
            assert out.scale_modifier == raster_settings.scale_modifier

        if out.sh_degree < 0:
            out.sh_degree = raster_settings.sh_degree
        else:
            assert out.sh_degree == raster_settings.sh_degree

        if out.H <= 0:
            out.H = raster_settings.image_height
        else:
            assert out.H == raster_settings.image_height

        if out.W <= 0:
            out.W = raster_settings.image_width
        else:
            assert out.W == raster_settings.image_width

        # insert required data
        out.debug = out.debug or raster_settings.debug
        out.viewmatrices.append(raster_settings.viewmatrix.contiguous())
        out.projmatrices.append(raster_settings.projmatrix.contiguous())
        out.camposes.append(raster_settings.campos.contiguous())
        out.tanfovxs.append(raster_settings.tanfovx)
        out.tanfovys.append(raster_settings.tanfovy)
        out.cxs.append(raster_settings.cx)
        out.cys.append(raster_settings.cy)
        out.geomBuffers.append(geomBuffer.contiguous())
        out.binningBuffers.append(binningBuffer.contiguous())
        out.imgBuffers.append(imgBuffer.contiguous())
        out.num_rendered_list.append(num_rendered)
        out.n_contrib_vol_rend.append(n_contrib_vol_rend)
        out.is_gaussian_hit.append(is_gaussian_hit)

    # stack some attrs to tensors
    out.tanfovxs = torch.tensor(out.tanfovxs, device=device, dtype=torch.float32)
    out.tanfovys = torch.tensor(out.tanfovys, device=device, dtype=torch.float32)
    out.cxs = torch.tensor(out.cxs, device=device, dtype=torch.float32)
    out.cys = torch.tensor(out.cys, device=device, dtype=torch.float32)
    out.camposes = torch.stack(out.camposes)

    # stack residuals & weights
    # we perform flatten() on stacked tensors of shape (C, num_images, H, W)
    out.residuals = torch.stack(residuals_list, dim=1).flatten()
    if compute_huber_weights:
        out.weights = torch.stack(weights_list, dim=1).flatten()
    if compute_ssim_weights:
        out.residuals_ssim = torch.stack(residuals_ssim_list, dim=1).flatten()
        out.weights_ssim = torch.stack(weights_ssim_list, dim=1).flatten()

    return out


def eval_jtf_and_get_sparse_jacobian(
    gaussians: GaussianModel,
    forward_output: RenderedImageAndBackwardValues,
    timing_dict: Dict[str, float] = None,
    use_double_precision: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Any]:

    return GaussianRasterizer.eval_jtf_and_get_sparse_jacobian(
        means3D=gaussians.get_xyz,
        unactivated_rotations=gaussians._rotation,
        unactivated_scales=gaussians._scaling,
        unactivated_opacities=gaussians._opacity,
        shs=gaussians.get_active_features_rest,
        params=gaussians.get_reordered_params(with_SH=False),
        forward_output=forward_output,
        use_double_precision=use_double_precision,
        timing_dict=timing_dict
    )


def apply_jtj(
    x: torch.Tensor,
    x_resorted: torch.Tensor,
    sparse_jacobians: List[torch.Tensor],
    index_map: List[torch.Tensor],
    per_gaussian_caches: List[torch.Tensor],
    data,
    timing_dict: Dict[str, float] = None,
    segments: List[torch.Tensor] = None,
    segments_to_gaussians_list: List[torch.Tensor] = None,
    num_gaussians_in_block: List[torch.Tensor] = None,
    block_offset_in_segments: List[torch.Tensor] = None
) -> torch.Tensor:

    return GaussianRasterizer.apply_jtj(
        x=x,
        x_resorted=x_resorted,
        sparse_jacobians=sparse_jacobians,
        index_map=index_map,
        per_gaussian_caches=per_gaussian_caches,
        data=data,
        segments=segments,
        segments_to_gaussians_list=segments_to_gaussians_list,
        num_gaussians_in_block=num_gaussians_in_block,
        block_offset_in_segments=block_offset_in_segments,
        timing_dict=timing_dict
    )


def calc_preconditioner(
    sparse_jacobians: List[torch.Tensor],
    index_map: List[torch.Tensor],
    per_gaussian_caches: List[torch.Tensor],
    data,
    timing_dict: Dict[str, float] = None,
    segments: List[torch.Tensor] = None,
    segments_to_gaussians_list: List[torch.Tensor] = None,
    num_gaussians_in_block: List[torch.Tensor] = None,
    block_offset_in_segments: List[torch.Tensor] = None
) -> torch.Tensor:

    return GaussianRasterizer.calc_preconditioner(
        sparse_jacobians=sparse_jacobians,
        index_map=index_map,
        per_gaussian_caches=per_gaussian_caches,
        data=data,
        segments=segments,
        segments_to_gaussians_list=segments_to_gaussians_list,
        num_gaussians_in_block=num_gaussians_in_block,
        block_offset_in_segments=block_offset_in_segments,
        timing_dict=timing_dict
    )


def apply_j(
    x: torch.Tensor,
    x_resorted: torch.Tensor,
    sparse_jacobians: List[torch.Tensor],
    index_map: List[torch.Tensor],
    per_gaussian_caches: List[torch.Tensor],
    data,
    timing_dict: Dict[str, float] = None,
    segments: List[torch.Tensor] = None,
    segments_to_gaussians_list: List[torch.Tensor] = None,
    num_gaussians_in_block: List[torch.Tensor] = None,
    block_offset_in_segments: List[torch.Tensor] = None
) -> torch.Tensor:
    return GaussianRasterizer.apply_j(
        x=x,
        x_resorted=x_resorted,
        sparse_jacobians=sparse_jacobians,
        index_map=index_map,
        per_gaussian_caches=per_gaussian_caches,
        data=data,
        segments=segments,
        segments_to_gaussians_list=segments_to_gaussians_list,
        num_gaussians_in_block=num_gaussians_in_block,
        block_offset_in_segments=block_offset_in_segments,
        timing_dict=timing_dict
    )
