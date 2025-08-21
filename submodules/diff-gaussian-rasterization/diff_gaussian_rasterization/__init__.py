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

from typing import NamedTuple, List, Tuple, Union, Iterable
import torch.nn as nn
import torch
from . import _C
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    cx : float
    cy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    prepare_for_gsgn_backward: bool
    debug : bool


@dataclass
class RenderedImageAndBackwardValues:
    num_images: int = 0
    H: int = 0
    W: int = 0
    bg: torch.Tensor = None  # same across all raster_settings
    scale_modifier: float = -1.0  # same across all raster_settings
    sh_degree: int = -1 # same across all raster_settings
    debug: bool = False # same across all raster_settings
    viewmatrices: List = field(default_factory=lambda: [])
    projmatrices: List = field(default_factory=lambda: [])
    camposes: List = field(default_factory=lambda: [])
    tanfovxs: List = field(default_factory=lambda: [])
    tanfovys: List = field(default_factory=lambda: [])
    cxs: List = field(default_factory=lambda: [])
    cys: List = field(default_factory=lambda: [])
    geomBuffers: List = field(default_factory=lambda: [])
    binningBuffers: List = field(default_factory=lambda: [])
    imgBuffers: List = field(default_factory=lambda: [])
    num_rendered_list: List = field(default_factory=lambda: [])
    n_contrib_vol_rend: List = field(default_factory=lambda: [])
    is_gaussian_hit: List = field(default_factory=lambda: [])
    residuals: torch.Tensor = None
    weights: torch.Tensor = None
    residuals_ssim: torch.Tensor = None
    weights_ssim: torch.Tensor = None


def get_list_size_in_bytes(x: List[torch.Tensor]) -> float:
    if x is None:
        return 0
    return sum([xi.numel() * xi.element_size() for xi in x])


def get_forward_output_size(output: RenderedImageAndBackwardValues) -> float:
    size_in_bytes = 0

    size_in_bytes += get_list_size_in_bytes(output.viewmatrices)
    size_in_bytes += get_list_size_in_bytes(output.projmatrices)
    size_in_bytes += get_list_size_in_bytes(output.camposes)
    size_in_bytes += get_list_size_in_bytes(output.tanfovxs)
    size_in_bytes += get_list_size_in_bytes(output.tanfovys)
    size_in_bytes += get_list_size_in_bytes(output.cxs)
    size_in_bytes += get_list_size_in_bytes(output.cys)
    size_in_bytes += get_list_size_in_bytes(output.geomBuffers)
    size_in_bytes += get_list_size_in_bytes(output.binningBuffers)
    size_in_bytes += get_list_size_in_bytes(output.imgBuffers)
    size_in_bytes += get_list_size_in_bytes(output.n_contrib_vol_rend)
    size_in_bytes += get_list_size_in_bytes(output.is_gaussian_hit)

    size_in_bytes += output.bg.numel() * output.bg.element_size()

    # do not include residuals since they live on CPU for the PCG() iterations
    # do include weights since they are used in the PCG() iterations
    if output.weights is not None:
        size_in_bytes += output.weights.numel() * output.weights.element_size()
    if output.weights_ssim is not None:
        size_in_bytes += output.weights_ssim.numel() * output.weights_ssim.element_size()

    return size_in_bytes


@contextmanager
def measure_time(name: str, out_dict: Dict[str, float], additive: bool = False, maximum: bool = False):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        torch.cuda.synchronize()
        if out_dict is not None:
            elapsed = start.elapsed_time(end)
            if additive and name in out_dict:
                out_dict[name] += elapsed
            else:
                if maximum and name in out_dict:
                    out_dict[name] = max(elapsed, out_dict[name])
                else:
                    out_dict[name] = elapsed


def cpu_deep_copy_gsgn_data_spec(gsgn_data_spec: _C.GSGNDataSpec):
    vars = {
        "background": gsgn_data_spec.background,
        "means3D": gsgn_data_spec.means3D,
        "scales": gsgn_data_spec.scales,
        "rotations": gsgn_data_spec.rotations,
        "scale_modifier": gsgn_data_spec.scale_modifier,
        "viewmatrix": gsgn_data_spec.viewmatrix,
        "projmatrix": gsgn_data_spec.projmatrix,
        "tan_fovx": gsgn_data_spec.tan_fovx,
        "tan_fovy": gsgn_data_spec.tan_fovy,
        "cx": gsgn_data_spec.cx,
        "cy": gsgn_data_spec.cy,
        "sh": gsgn_data_spec.sh,
        "unactivated_opacity": gsgn_data_spec.unactivated_opacity,
        "unactivated_scales": gsgn_data_spec.unactivated_scales,
        "unactivated_rotations": gsgn_data_spec.unactivated_rotations,
        "degree": gsgn_data_spec.degree,
        "campos": gsgn_data_spec.campos,
        "geomBuffer": gsgn_data_spec.geomBuffer,
        "R": gsgn_data_spec.R,
        "binningBuffer": gsgn_data_spec.binningBuffer,
        "imgBuffers": gsgn_data_spec.imageBuffer,
        "use_double_precision": gsgn_data_spec.use_double_precision,
        "debug": gsgn_data_spec.debug,
        "residuals": gsgn_data_spec.residuals,
        "n_contrib_vol_rend_prefix_sum": gsgn_data_spec.n_contrib_vol_rend_prefix_sum,
        "num_sparse_gaussians": gsgn_data_spec.num_sparse_gaussians
    }

    for k, v in vars.items():
        if isinstance(v, torch.Tensor):
            vars[k] = v.cpu().clone()
        elif isinstance(v, Iterable):
            x = []
            for vi in v:
                if isinstance(vi, torch.Tensor):
                    x.append(vi.cpu().clone())
                else:
                    x.append(vi)
            vars[k] = x
        else:
            vars[k] = v

    return vars


def cpu_deep_copy(input: Iterable):
    return [
        item.cpu().clone() if isinstance(item, torch.Tensor) else
        cpu_deep_copy_gsgn_data_spec(item) if isinstance(item, _C.GSGNDataSpec) else
        item
        for item in input
    ]


def safe_call_fn(fn, args: Iterable, debug: bool):
    if debug:
        # Copy them before they can be corrupted
        cpu_args = cpu_deep_copy(args)
        try:
            out = fn(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_fw.dump")
            print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
            raise ex
    else:
        out = fn(*args)

    return out


def rasterize_forward_impl(
    means3D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    # Restructure arguments the way that the C++ lib expects them
    args = (
        raster_settings.bg, 
        means3D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        raster_settings.scale_modifier,
        cov3Ds_precomp,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.cx,
        raster_settings.cy,
        raster_settings.image_height,
        raster_settings.image_width,
        sh,
        raster_settings.sh_degree,
        raster_settings.campos,
        raster_settings.prefiltered,
        raster_settings.prepare_for_gsgn_backward,
        raster_settings.debug
    )

    # Invoke C++/CUDA rasterizer
    return safe_call_fn(_C.rasterize_gaussians, args, raster_settings.debug)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings: GaussianRasterizationSettings,
    ):
        num_rendered, n_contrib_vol_rend, is_gaussian_hit, color, radii, geomBuffer, binningBuffer, imgBuffer = rasterize_forward_impl(
            means3D=means3D,
            sh=sh,
            colors_precomp=colors_precomp,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3Ds_precomp=cov3Ds_precomp,
            raster_settings=raster_settings
        )

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, n_contrib_vol_rend, is_gaussian_hit

    @staticmethod
    def backward(ctx, grad_color, grad_radii, grad_n_contrib_vol_rend, grad_is_gaussian_hit):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy,
                raster_settings.cx,
                raster_settings.cy,
                grad_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = safe_call_fn(_C.rasterize_gaussians_backward, args, raster_settings.debug)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )


    @staticmethod
    def build_gsgn_data_spec(
        forward_output: RenderedImageAndBackwardValues,
        params: torch.Tensor,
        means3D: torch.Tensor = None,
        shs: torch.Tensor = None,
        unactivated_opacities: torch.Tensor = None,
        unactivated_scales: torch.Tensor = None,
        unactivated_rotations: torch.Tensor = None,
        use_double_precision: bool = False,
    ):
        """
        Generate object to pass to C++
        """
        gsgn_data_spec = _C.GSGNDataSpec()
        gsgn_data_spec.background = forward_output.bg
        gsgn_data_spec.params = params.contiguous()
        gsgn_data_spec.means3D = means3D.contiguous() if means3D is not None else None
        gsgn_data_spec.scale_modifier = forward_output.scale_modifier
        gsgn_data_spec.viewmatrix = forward_output.viewmatrices
        gsgn_data_spec.projmatrix = forward_output.projmatrices
        gsgn_data_spec.tan_fovx = forward_output.tanfovxs
        gsgn_data_spec.tan_fovy = forward_output.tanfovys
        gsgn_data_spec.cx = forward_output.cxs
        gsgn_data_spec.cy = forward_output.cys
        gsgn_data_spec.sh = shs.contiguous() if shs is not None else None
        gsgn_data_spec.unactivated_opacity = unactivated_opacities.contiguous() if unactivated_opacities is not None else None
        gsgn_data_spec.unactivated_scales = unactivated_scales.contiguous() if unactivated_scales is not None else None
        gsgn_data_spec.unactivated_rotations = unactivated_rotations.contiguous() if unactivated_rotations is not None else None
        gsgn_data_spec.degree = forward_output.sh_degree
        gsgn_data_spec.H = forward_output.H
        gsgn_data_spec.W = forward_output.W
        gsgn_data_spec.campos = forward_output.camposes
        gsgn_data_spec.R = forward_output.num_rendered_list
        gsgn_data_spec.binningBuffer = forward_output.binningBuffers
        gsgn_data_spec.imageBuffer = forward_output.imgBuffers
        gsgn_data_spec.use_double_precision = use_double_precision
        gsgn_data_spec.debug = forward_output.debug

        dummy = torch.empty([0], device=params.device, dtype=params.dtype)
        assert forward_output.residuals is not None
        gsgn_data_spec.residuals = forward_output.residuals
        gsgn_data_spec.weights = forward_output.weights if forward_output.weights is not None else dummy
        gsgn_data_spec.residuals_ssim = forward_output.residuals_ssim if forward_output.residuals_ssim is not None else dummy
        gsgn_data_spec.weights_ssim = forward_output.weights_ssim if forward_output.weights_ssim is not None else dummy

        # calculate n_contrib_vol_rend_prefix_sum
        x = torch.stack(forward_output.n_contrib_vol_rend, dim=0)  # (num_images, H, W)
        num_images, H, W = x.shape

        # pad the tensor s.t. it is dividable by the warp size
        def pad(x: torch.Tensor, warp_size: int, dim: int):
            pad_size = x.shape[dim] % warp_size
            if pad_size > 0:
                size = [1] * len(x.shape)
                size[dim] = warp_size - pad_size
                pad_tensor = torch.zeros(size, device=x.device, dtype=x.dtype)
                size = [i for i in x.shape]
                size[dim] = -1
                pad_tensor = pad_tensor.expand(*size)
                x = torch.cat([x, pad_tensor], dim=dim)
            return x

        x = pad(x, 16, 2)
        x = pad(x, 2, 1)

        # calculate the sum per warp. one warp is a 2x16 (HxW) patch in the image
        x = x.unfold(1, 2, 2).unfold(2, 16, 16)  # (num_images, num_warps_h, num_warps_w, 2, 16)
        x = x.reshape(num_images, x.shape[1], x.shape[2], -1)  # (num_images, num_warps_h, num_warps_w, 32)
        x = x.sum(dim=-1, dtype=x.dtype)  # (num_images, num_warps_h, num_warps_w)

        # prefix sum goes over the sum per warp
        x = x.view(num_images, -1)
        x = torch.cumsum(x, dim=1, dtype=x.dtype)

        # we construct the prefix_sum with torch.cumsum(data.n_contrib_vol_rend) which works like this: [10, 5, 8] --> [10, 15, 23]
        # the last elemnt is thus the total number of gaussians used in vol-rend of that image
        num_sparse_gaussians = [xi.item() for xi in x[:, -1]]

        # we want to use the prefix_sum as offset-map in the kernel. The first gaussians should start at index 0, not 10: [10, 15, 23] --> [0, 10, 15]
        x = x[:, :-1]
        zeros = torch.zeros_like(x[:, 0:1])
        x = torch.cat([zeros, x], dim=1)
        assert (x >= 0).all()

        gsgn_data_spec.n_contrib_vol_rend_prefix_sum = [xi for xi in x]
        gsgn_data_spec.num_sparse_gaussians = num_sparse_gaussians
        forward_output.n_contrib_vol_rend = None  # don't need it any more, save the memory

        # calculate map_visible_gaussians
        valid_mask = torch.stack(forward_output.is_gaussian_hit, dim=0)  # (num_images, P)
        x = torch.cumsum(valid_mask, dim=1, dtype=torch.int32)  # e.g. [1, 1, 0, 1, 1] -> [1, 2, 2, 3, 4]
        num_visible_gaussians = [xi.item() for xi in x[:, -1]]
        x = x - 1  # e.g. [1, 2, 2, 3, 4] -> [0, 1, 1, 2, 3]

        max_num_visible_gaussians = max(num_visible_gaussians)
        x[~valid_mask] = max_num_visible_gaussians  # e.g. [0, 1, 1, 2, 3] -> [0, 1, 4, 2, 3]
        map_cache_to_gaussians = torch.empty((x.shape[0], max_num_visible_gaussians + 1), dtype=torch.int32, device=x.device)
        src = torch.arange(0, x.shape[1], device=x.device, dtype=torch.int32)  # (0, 1, ..., P - 1)
        src = src[None].expand(x.shape[0], -1)
        map_cache_to_gaussians.scatter_(dim=1, index=x.to(torch.int64), src=src)
        map_cache_to_gaussians = [xi[:num_visible_gaussians[idx]] for idx, xi in enumerate(map_cache_to_gaussians)]
        gsgn_data_spec.map_cache_to_gaussians = map_cache_to_gaussians

        x[~valid_mask] = -1  # e.g. [0, 1, 1, 2, 3] -> [0, 1, -1, 2, 3]
        gsgn_data_spec.map_visible_gaussians = [xi for xi in x]
        gsgn_data_spec.num_visible_gaussians = num_visible_gaussians
        forward_output.is_gaussian_hit = None  # don't need it any more, save the memory

        # filter geometry buffers
        filtered_geomBuffers = safe_call_fn(_C.filter_reordered_geometry_buffer, [forward_output.geomBuffers, map_cache_to_gaussians, num_visible_gaussians], forward_output.debug)
        gsgn_data_spec.geomBuffer = filtered_geomBuffers
        forward_output.geomBuffers = filtered_geomBuffers

        return gsgn_data_spec

    @staticmethod
    def subsample_data(in_gsgn_data_spec: _C.GSGNDataSpec, indices) -> _C.GSGNDataSpec:
        gsgn_data_spec = _C.GSGNDataSpec()
        gsgn_data_spec.params = in_gsgn_data_spec.params
        gsgn_data_spec.background = in_gsgn_data_spec.background
        gsgn_data_spec.means3D = in_gsgn_data_spec.means3D
        gsgn_data_spec.scale_modifier = in_gsgn_data_spec.scale_modifier
        gsgn_data_spec.viewmatrix = [in_gsgn_data_spec.viewmatrix[i] for i in indices]
        gsgn_data_spec.projmatrix = [in_gsgn_data_spec.projmatrix[i] for i in indices]
        gsgn_data_spec.tan_fovx = in_gsgn_data_spec.tan_fovx[indices]
        gsgn_data_spec.tan_fovy = in_gsgn_data_spec.tan_fovy[indices]
        gsgn_data_spec.cx = in_gsgn_data_spec.cx[indices]
        gsgn_data_spec.cy = in_gsgn_data_spec.cy[indices]
        gsgn_data_spec.sh = in_gsgn_data_spec.sh
        gsgn_data_spec.unactivated_opacity = in_gsgn_data_spec.unactivated_opacity
        gsgn_data_spec.unactivated_scales = in_gsgn_data_spec.unactivated_scales
        gsgn_data_spec.unactivated_rotations = in_gsgn_data_spec.unactivated_rotations
        gsgn_data_spec.degree = in_gsgn_data_spec.degree
        gsgn_data_spec.H = in_gsgn_data_spec.H
        gsgn_data_spec.W = in_gsgn_data_spec.W
        gsgn_data_spec.campos = in_gsgn_data_spec.campos[indices]
        gsgn_data_spec.geomBuffer = [in_gsgn_data_spec.geomBuffer[i] for i in indices]
        gsgn_data_spec.R = [in_gsgn_data_spec.R[i] for i in indices]
        gsgn_data_spec.binningBuffer = [in_gsgn_data_spec.binningBuffer[i] for i in indices]
        gsgn_data_spec.imageBuffer = [in_gsgn_data_spec.imageBuffer[i] for i in indices]
        gsgn_data_spec.use_double_precision = in_gsgn_data_spec.use_double_precision
        gsgn_data_spec.debug = in_gsgn_data_spec.debug
        gsgn_data_spec.n_contrib_vol_rend_prefix_sum = [in_gsgn_data_spec.n_contrib_vol_rend_prefix_sum[i] for i in indices] if len(in_gsgn_data_spec.n_contrib_vol_rend_prefix_sum) > 0 else in_gsgn_data_spec.n_contrib_vol_rend_prefix_sum
        gsgn_data_spec.num_sparse_gaussians = [in_gsgn_data_spec.num_sparse_gaussians[i] for i in indices]
        gsgn_data_spec.map_visible_gaussians = [in_gsgn_data_spec.map_visible_gaussians[i] for i in indices]
        gsgn_data_spec.map_cache_to_gaussians = [in_gsgn_data_spec.map_cache_to_gaussians[i] for i in indices]
        gsgn_data_spec.num_visible_gaussians = [in_gsgn_data_spec.num_visible_gaussians[i] for i in indices]

        # do not subsample residuals/weights (it would be possible...), because subsampling is only used in sorting kernels and those dont need these infos anyways
        dummy = torch.empty([0], device=gsgn_data_spec.params.device, dtype=gsgn_data_spec.params.dtype)
        gsgn_data_spec.residuals = dummy
        gsgn_data_spec.weights = dummy
        gsgn_data_spec.residuals_ssim = dummy
        gsgn_data_spec.weights_ssim = dummy

        return gsgn_data_spec
    
    @staticmethod
    def eval_jtf_and_get_sparse_jacobian(
        params: torch.Tensor,
        means3D: torch.Tensor = None,
        shs: torch.Tensor = None,
        unactivated_opacities: torch.Tensor = None,
        unactivated_scales: torch.Tensor = None,
        unactivated_rotations: torch.Tensor = None,
        forward_output: RenderedImageAndBackwardValues = None,
        use_double_precision: bool = False,
        timing_dict: Dict[str, float] = None):

        # check if we need to render or if the outputs are provided as input
        if forward_output is None:
            raise Exception('Please provide forward output!')

        with measure_time("eval_jtf_and_get_sparse_jacobian", timing_dict, maximum=True):
            data = GaussianRasterizer.build_gsgn_data_spec(
                params=params,
                means3D=means3D,
                shs=shs,
                unactivated_opacities=unactivated_opacities,
                unactivated_scales=unactivated_scales,
                unactivated_rotations=unactivated_rotations,
                forward_output=forward_output,
                use_double_precision=use_double_precision,
            )

            # fuse residuals/weights into one tensor --> less global memory reads in the kernel
            if forward_output.weights is not None:
                data.residuals = data.residuals * data.weights
            if forward_output.residuals_ssim is not None and forward_output.weights_ssim is not None:
                data.residuals += data.residuals_ssim * data.weights_ssim

            # Invoke C++/CUDA rasterizer
            r, sparse_jacobians, index_maps, per_gaussian_caches = safe_call_fn(_C.eval_jtf_and_get_sparse_jacobian, [data], forward_output.debug)

            # move residuals to CPU --> only needed for update after PCG() finished
            forward_output.residuals = forward_output.residuals.to(device='cpu', non_blocking=False)  # True
            if forward_output.residuals_ssim is not None:
                forward_output.residuals_ssim = forward_output.residuals_ssim.to(device='cpu', non_blocking=False)  # True

            # no other kernels after eval_jtf need these any more, save memory
            dummy = torch.empty([0], device=params.device, dtype=params.dtype)
            data.residuals = dummy
            data.n_contrib_vol_rend_prefix_sum = []
            data.residuals_ssim = dummy

            # remove binningBuffer and imgBuffer data, is not needed anymore, save memory
            dummy = torch.empty([len(forward_output.binningBuffers), 0], device=forward_output.binningBuffers[0].device, dtype=forward_output.binningBuffers[0].dtype)
            dummy = [xi for xi in dummy]
            forward_output.binningBuffers = dummy
            forward_output.imgBuffers = dummy
            data.binningBuffer = dummy
            data.imageBuffer = dummy

        return r, sparse_jacobians, index_maps, per_gaussian_caches, data

    @staticmethod
    def apply_jtj(
        x: torch.Tensor,
        x_resorted: torch.Tensor,
        sparse_jacobians: List[torch.Tensor],
        index_map: List[torch.Tensor],
        per_gaussian_caches: List[torch.Tensor],
        data: _C.GSGNDataSpec,
        segments: List[torch.Tensor] = None,
        segments_to_gaussians_list: List[torch.Tensor] = None,
        num_gaussians_in_block: List[torch.Tensor] = None,
        block_offset_in_segments: List[torch.Tensor] = None,
        timing_dict: Dict[str, float] = None,
        debug: bool = False):

        with measure_time("apply_jtj", timing_dict):
            # Invoke C++/CUDA rasterizer
            g = safe_call_fn(_C.apply_jtj, [data, x, x_resorted, sparse_jacobians, index_map, per_gaussian_caches, segments, segments_to_gaussians_list, num_gaussians_in_block, block_offset_in_segments], debug)

        return g

    @staticmethod
    def calc_preconditioner(
        sparse_jacobians: List[torch.Tensor],
        index_map: List[torch.Tensor],
        per_gaussian_caches: List[torch.Tensor],
        data: _C.GSGNDataSpec,
        segments: List[torch.Tensor] = None,
        segments_to_gaussians_list: List[torch.Tensor] = None,
        num_gaussians_in_block: List[torch.Tensor] = None,
        block_offset_in_segments: List[torch.Tensor] = None,
        timing_dict: Dict[str, float] = None,
        debug: bool = False):

        with measure_time("calc_preconditioner", timing_dict):
            # Invoke C++/CUDA rasterizer
            M = safe_call_fn(_C.calc_preconditioner, [data, sparse_jacobians, index_map, per_gaussian_caches, segments, segments_to_gaussians_list, num_gaussians_in_block, block_offset_in_segments], debug)

        return M

    @staticmethod
    def apply_j(
        x: torch.Tensor,
        x_resorted: torch.Tensor,
        sparse_jacobians: List[torch.Tensor],
        index_map: List[torch.Tensor],
        per_gaussian_caches: List[torch.Tensor],
        data: _C.GSGNDataSpec,
        segments: List[torch.Tensor] = None,
        segments_to_gaussians_list: List[torch.Tensor] = None,
        num_gaussians_in_block: List[torch.Tensor] = None,
        block_offset_in_segments: List[torch.Tensor] = None,
        timing_dict: Dict[str, float] = None,
        debug: bool = False):

        with measure_time("apply_j", timing_dict):
            # Invoke C++/CUDA rasterizer
            jx = safe_call_fn(_C.apply_j, [data, x, x_resorted, sparse_jacobians, index_map, per_gaussian_caches, segments, segments_to_gaussians_list, num_gaussians_in_block, block_offset_in_segments], debug)

        return jx

    @staticmethod
    def sort_sparse_jacobians(
        sparse_jacobians: List[torch.Tensor],
        indices_map: List[torch.Tensor],
        data: _C.GSGNDataSpec,
        timing_dict: Dict[str, float] = None,
        debug: bool = False):

        with measure_time("sort_sparse_jacobians", timing_dict):
            # Invoke C++/CUDA rasterizer
            sorted_sparse_jacobians = safe_call_fn(_C.sort_sparse_jacobians, [data, sparse_jacobians, indices_map], debug)

        return sorted_sparse_jacobians

    @staticmethod
    def extract_gaussian_parameters(x: torch.Tensor, data: _C.GSGNDataSpec, make_contiguous: bool = True):
        # num_floats_per_gaussian = 11 + data.M * 3
        # num_gaussians = x.shape[0] // num_floats_per_gaussian
        # assert(num_gaussians == data.P)
        # x = x.view(data.P, num_floats_per_gaussian)
        # x_xyz = x[:, 0:3].contiguous()
        # x_rotation = x[:, 3:7].contiguous()
        # x_scaling = x[:, 7:10].contiguous()
        # x_opacity = x[:, 10:11].contiguous()
        # x_features = x[:, 11:]  # (P, M*3)
        # x_features = x_features.view(data.P, 3, data.M).permute(0, 2, 1)  # (P, M, 3)
        # x_features_dc = x_features[:, 0:1, :]
        # x_features_rest = x_features[:, 1:, :]

        # extract xyz
        a = data.offset_xyz
        b = data.offset_scales
        x_xyz = x[a:b]
        # we wrote in the shape (3, P), but need (P, 3)
        x_xyz = x_xyz.view(3, data.P).permute(1, 0)

        a = b
        b = data.offset_rotations
        x_scaling = x[a:b]
        # we wrote in the shape (3, P), but need (P, 3)
        x_scaling = x_scaling.view(3, data.P).permute(1, 0)

        a = b
        b = data.offset_opacity
        x_rotation = x[a:b]
        # we wrote in the shape (4, P), but need (P, 4)
        x_rotation = x_rotation.view(4, data.P).permute(1, 0)

        a = b
        b = data.offset_features_dc
        x_opacity = x[a:b]
        # we wrote in the shape (P), but need (P, 1)
        x_opacity = x_opacity.unsqueeze(-1)

        a = b
        b = data.offset_features_rest
        x_features_dc = x[a:b]
        # we wrote in the shape (3, P), but need (P, 1, 3)
        x_features_dc = x_features_dc.view(3, data.P).permute(1, 0).unsqueeze(1)

        a = b
        x_features_rest = x[a:]
        # we wrote in the shape (M-1, 3, P), but need (P, M-1, 3)
        x_features_rest = x_features_rest.view(data.M - 1, 3, data.P).permute(2, 0, 1)

        x_dict = {
            "xyz": x_xyz,
            "scaling": x_scaling,
            "rotation": x_rotation,
            "opacity": x_opacity,
            "features_dc": x_features_dc,
            "features_rest": x_features_rest
        }

        if make_contiguous:
            x_dict = {k: v.contiguous() for k, v in x_dict.items()}

        return x_dict
