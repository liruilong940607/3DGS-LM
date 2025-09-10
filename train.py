#
# This file is extended from the original train.py file to call the LM optimizer.
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
import json
import time
import os
import random
import math

import torch
from random import randint
from typing import Dict, List, Union
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import (
    render,
    network_gui,
    render_and_add_to_residual_norm,
    render_all_images_and_backward,
    eval_jtf_and_get_sparse_jacobian,
    apply_jtj,
    apply_j,
    calc_preconditioner,
    RenderedImageAndBackwardValues
)
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state
import uuid
from tqdm.auto import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, GSGNParams
from cg_batch import cg_batch
from diff_gaussian_rasterization import GaussianRasterizer, get_forward_output_size, measure_time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from render import render_sets
from metrics import evaluate


@torch.no_grad()
def get_residual_norm(x: torch.Tensor):
    norm = 0

    norm += (x ** 2).sum()
    # norm += x.abs().sum()

    return 0.5 * norm


def linear_solve_pcg_fused(
    gaussians: GaussianModel,
    viewpoint_stack: List[Camera],
    opt: OptimizationParams,
    pipe: PipelineParams,
    background: torch.Tensor,
    gsgn: GSGNParams,
    trust_region_radius: float = 1.0,
    timing_dict: Dict[str, float] = None,
    forward_output: RenderedImageAndBackwardValues = None):

    with measure_time("render_all_images", timing_dict, additive=True):
        if forward_output is None:
            # render once and cache outputs for the following computations
            forward_output = render_all_images_and_backward(
                gaussians=gaussians,
                viewpoint_stack=viewpoint_stack,
                opt=opt,
                pipe=pipe,
                background=background,
                compute_huber_weights=gsgn.compute_huber_weights,
                huber_c=gsgn.huber_c,
                compute_ssim_weights=gsgn.compute_ssim_weights,
                ssim_residual_scale_factor=gsgn.ssim_residual_scale_factor
            )

        # calc mean gaussians/ray for logging
        mean_n_contrib_per_pixel = 0

    with measure_time("pcg", timing_dict):
        with measure_time("eval_jtf", timing_dict, additive=True):
            # b = J.t() @ F
            b, sparse_jacobians, index_maps, per_gaussian_caches, data = eval_jtf_and_get_sparse_jacobian(
                gaussians=gaussians,
                forward_output=forward_output,
                timing_dict=timing_dict,
                use_double_precision=gsgn.use_double_precision,
            )

            with measure_time("sort_index_maps", timing_dict, additive=True):
                segment_list = []
                segments_to_gaussians_list = []
                num_gaussians_in_block_list = []
                block_offset_in_segments_list = []
                for i in range(len(index_maps)):
                    with measure_time("sort_preprocess", timing_dict, additive=True):
                        m = index_maps[i]
                        half = m.numel() // 2
                        gaussian_ids = m[:half]
                        ray_ids = m[half:]

                        sort_keys = gaussian_ids.to(torch.int64) * data.H * data.W * data.num_images + ray_ids.to(torch.int64)
                        _, indices = torch.sort(sort_keys)
                        sorted_gaussians = gaussian_ids[indices]
                        sorted_ray_ids = ray_ids[indices]

                        # don't need sorted_gaussians anymore, segments already contains it
                        index_maps[i] = sorted_ray_ids

                    with measure_time("sort_kernel", timing_dict, additive=True):
                        data_sub = GaussianRasterizer.subsample_data(data, [i])
                        sparse_jacobians[i] = GaussianRasterizer.sort_sparse_jacobians(
                            sparse_jacobians=[sparse_jacobians[i]],
                            indices_map=[indices],
                            data=data_sub,
                            timing_dict=timing_dict
                        )[0]

                    with measure_time("sort_postprocess_intermediate", timing_dict, additive=True):
                        threads_per_block = 128
                        num_blocks = math.ceil(len(sorted_gaussians) / threads_per_block)
                        segments = torch.nonzero(sorted_gaussians[1:] - sorted_gaussians[:-1]).int().flatten() + 1
                        block_borders = torch.arange(start=0, end=num_blocks, dtype=segments.dtype, device=segments.device) * threads_per_block
                        segments = torch.cat([block_borders, segments])
                        segments = torch.unique(segments, sorted=True) # necessary because we want to sort it again and there could be duplicates now (if segment starts already at block border)
                        segments_to_block = segments // threads_per_block
                        _, num_gaussians_in_block = torch.unique_consecutive(segments_to_block, return_counts=True)
                        num_gaussians_in_block = num_gaussians_in_block.int()
                        assert len(num_gaussians_in_block) == num_blocks
                        block_offset_in_segments = torch.cumsum(num_gaussians_in_block, dim=0, dtype=num_gaussians_in_block.dtype)[:-1]
                        block_offset_in_segments = torch.cat([torch.zeros_like(block_offset_in_segments[0:1]), block_offset_in_segments])
                        segment_list.append(segments)
                        segments_to_gaussians_list.append(sorted_gaussians[segments])  # save the actual gaussian global_id
                        num_gaussians_in_block_list.append(num_gaussians_in_block)
                        block_offset_in_segments_list.append(block_offset_in_segments)

            del sort_keys
            del indices
            del sorted_gaussians
            del sorted_ray_ids
            del threads_per_block
            del num_blocks
            del segments
            del block_borders
            del segments_to_block
            del num_gaussians_in_block
            del block_offset_in_segments
            del data_sub
            del gaussian_ids
            del ray_ids
            del m
            del half
            del i

        with measure_time("build_M_CTC", timing_dict):
            # M = diag(JTJ)
            M = calc_preconditioner(
                sparse_jacobians=sparse_jacobians,
                index_map=index_maps,
                per_gaussian_caches=per_gaussian_caches,
                data=data,
                timing_dict=timing_dict,
                segments=segment_list,
                segments_to_gaussians_list=segments_to_gaussians_list,
                num_gaussians_in_block=num_gaussians_in_block_list,
                block_offset_in_segments=block_offset_in_segments_list
            )

            CTC = torch.clamp(M, gsgn.min_lm_diagonal, gsgn.max_lm_diagonal)
            M = 1.0 / (M + CTC)
            M = M.unsqueeze(-1)
            CTC = (1.0 / trust_region_radius) * CTC.unsqueeze(-1)

            size_dict = {
                "sparse_jacobians": sum([x.numel() * x.element_size() for x in sparse_jacobians]),
                "index_maps": sum([x.numel() * x.element_size() for x in index_maps]),
                "per_gaussian_caches": sum([x.numel() * x.element_size() for x in per_gaussian_caches]),
                "M_vec": M.numel() * M.element_size(),
                "CTC_vec": CTC.numel() * CTC.element_size(),
                "b_vec": b.numel() * b.element_size(),
                "forward_output": get_forward_output_size(forward_output),
                "segments": sum([x.numel() * x.element_size() for x in segment_list]),
                "segments_to_gaussians": sum([x.numel() * x.element_size() for x in segments_to_gaussians_list]),
                "num_gaussians_in_block": sum([x.numel() * x.element_size() for x in num_gaussians_in_block_list]),
                "block_offset_in_segments": sum([x.numel() * x.element_size() for x in block_offset_in_segments_list]),
                "map_cache_to_gaussians": sum([x.numel() * x.element_size() for x in data.map_cache_to_gaussians]),
                "map_visible_gaussians": sum([x.numel() * x.element_size() for x in data.map_visible_gaussians]),
                "n_contrib_vol_rend_prefix_sum": sum([x.numel() * x.element_size() for x in data.n_contrib_vol_rend_prefix_sum]),
                "params": data.params.numel() * data.params.element_size(),
            }
            total_size_in_byte = sum(size_dict.values())
            perc_size_dict = {k: v / total_size_in_byte for k, v in size_dict.items()}
            total_size_in_gb = total_size_in_byte / (1024 ** 3)
            # print(perc_size_dict)

        def M_bmm(X):
            with measure_time("M_bmm", timing_dict, additive=True):
                return (M * X[0]).unsqueeze(0)

        @torch.enable_grad()
        def A_bmm(X):
            with measure_time("A_bmm", timing_dict, additive=True):
                x = X.squeeze()
                x_resorted = gaussians.get_resorted_vec(x)
                g = apply_jtj(
                    x=x,
                    x_resorted=x_resorted,
                    sparse_jacobians=sparse_jacobians,
                    index_map=index_maps,
                    per_gaussian_caches=per_gaussian_caches,
                    data=data,
                    timing_dict=timing_dict,
                    segments=segment_list,
                    segments_to_gaussians_list=segments_to_gaussians_list,
                    num_gaussians_in_block=num_gaussians_in_block_list,
                    block_offset_in_segments=block_offset_in_segments_list
                )

                # use x_resorted to in_place calculate CTC * X[0]
                x_resorted *= 0
                x_resorted += CTC.squeeze()
                x_resorted *= X.squeeze()

                g += x_resorted
                g = g.unsqueeze(0).unsqueeze(-1)
            return g

        with measure_time("cg_batch", timing_dict):
            x, info = cg_batch(
                A_bmm=A_bmm,
                B=b.unsqueeze(0).unsqueeze(-1),
                M_bmm=M_bmm,
                maxiter=gsgn.pcg_max_iter,
                atol=gsgn.pcg_atol,
                rtol=gsgn.pcg_rtol,
                gradient_descent_every=gsgn.pcg_gradient_descent_every,
                explicit_residual_every=gsgn.pcg_explicit_residual_every,
                verbose=gsgn.pcg_verbose
            )

        x = x.squeeze().float()
        val_max = x.abs().max()
        max_grad_norm = gsgn.max_grad_norm
        scale_factor = min(1.0, max_grad_norm / val_max)
        x = scale_factor * x

    return {
        "x": x,
        "data": data,
        "forward_output": forward_output,
        "sparse_jacobians": sparse_jacobians,
        "index_maps": index_maps,
        "per_gaussian_caches": per_gaussian_caches,
        "segments": segment_list,
        "segments_to_gaussians": segments_to_gaussians_list,
        "num_gaussians_in_block": num_gaussians_in_block_list,
        "block_offset_in_segments": block_offset_in_segments_list,
        "diagJTJ": (1.0 / M.squeeze()) - trust_region_radius * CTC.squeeze(),
        "log_info": {
            "pcg_info": info,
            "mean_n_contrib_per_pixel": mean_n_contrib_per_pixel,
            "total_size_in_gb": total_size_in_gb,
        }
    }


def get_subsample_indices(indices: List[int], n_images: int, gn: GSGNParams) -> List[int]:
    selected_indices = []
    n_selectable = len(indices)

    # edge case: not enough frames remaining in indices --> take all remaining and fill indices back up
    remaining_indices = []
    if n_selectable <= gn.image_subsample_size:
        remaining_indices = [i for i in indices]
        selected_indices.extend(remaining_indices)
        indices.clear()
        indices.extend([i for i in range(n_images) if i not in remaining_indices])
        n_selectable = len(indices)
        if gn.image_subsample_frame_selection_mode == "random":
            random.shuffle(indices)

    # add to selected_indices
    n_to_sample = gn.image_subsample_size - len(selected_indices)
    if n_to_sample > 0:
        if gn.image_subsample_frame_selection_mode == "random":
            # indices are already randomly shuffled, just select sequentially + remove them from the list of all indices
            selected_indices.extend(indices[:n_to_sample])
            del indices[:n_to_sample]
        elif gn.image_subsample_frame_selection_mode == "strided":
            stride = n_selectable // n_to_sample
            x = indices[::stride][:n_to_sample]
            selected_indices.extend(x)
            for val in x:
                indices.remove(val)
        else:
            raise NotImplementedError("image_subsample_frame_selection_mode", gn.image_subsample_frame_selection_mode)

    # edge case: add back remaining_indices to indices s.t. they are available next iteration
    if len(remaining_indices) > 0:
        indices.extend(remaining_indices)
        if gn.image_subsample_frame_selection_mode == "random":
            random.shuffle(indices)

    assert len(selected_indices) == gn.image_subsample_size, f"failed to subsample {gn.image_subsample_size} images, only got {len(selected_indices)}"
    return selected_indices


@torch.no_grad()
def lm_step(
    gaussians: GaussianModel,
    viewpoint_stack: List[Camera],
    viewpoint_stack_indices: List[int],
    opt: OptimizationParams,
    pipe: PipelineParams,
    background: torch.Tensor,
    gsgn: GSGNParams,
    iteration: int,
    trust_region_radius: float = 1.0,
    radius_decrease_factor: float = 2.0,
    timing_dict: Dict[str, float] = None,
    forward_output: RenderedImageAndBackwardValues = None,
):
    with measure_time("linear_solve", timing_dict):
        use_all_images_at_once = gsgn.image_subsample_n_iters <= 1 and gsgn.image_subsample_size <= 0
        if gsgn.image_subsample_n_iters <= 1:
            # get images
            if gsgn.image_subsample_size >= 0:
                subsample_indices = get_subsample_indices(indices=viewpoint_stack_indices, n_images=len(viewpoint_stack), gn=gsgn)
                lm_step_images = [viewpoint_stack[i] for i in subsample_indices]
                forward_output = None
            else:
                lm_step_images = [viewpoint_stack[i] for i in viewpoint_stack_indices]

            # do one PCG fit over all lm_step_images
            out_dict = linear_solve_pcg_fused(
                gaussians=gaussians,
                viewpoint_stack=lm_step_images,
                opt=opt,
                pipe=pipe,
                background=background,
                gsgn=gsgn,
                trust_region_radius=trust_region_radius,
                timing_dict=timing_dict,
                forward_output=forward_output
            )
            forward_output = out_dict["forward_output"]
            x = out_dict["x"]
        else:
            forward_output = None
            x_list = []
            diag_JTJ_list = []
            used_images = []
            for idx in range(gsgn.image_subsample_n_iters):
                subsample_indices = get_subsample_indices(indices=viewpoint_stack_indices, n_images=len(viewpoint_stack), gn=gsgn)
                imgs = [viewpoint_stack[i] for i in subsample_indices]
                out_dict = linear_solve_pcg_fused(
                    gaussians=gaussians,
                    viewpoint_stack=imgs,
                    opt=opt,
                    pipe=pipe,
                    background=background,
                    gsgn=gsgn,
                    trust_region_radius=trust_region_radius,
                    timing_dict=timing_dict,
                    forward_output=None
                )
                x_list.append(out_dict["x"])
                diag_JTJ_list.append(out_dict["diagJTJ"])
                used_images.extend(imgs)
                if idx < gsgn.image_subsample_n_iters - 1:
                    out_dict = None
                    torch.cuda.empty_cache()

            # use all images for the remaining line-search + cost_norm calculations
            lm_step_images = used_images
            last_batch_images = imgs
            last_batch_forward_output = out_dict["forward_output"]

            # combine multiple PCG solutions into one final solution for this lm step
            if gsgn.average_pcg_mode == "diag_jtj":
                # implementation that does not require to create new intermediate tensors
                denom = diag_JTJ_list[0].clone()
                denom += 1e-12
                for i in range(len(diag_JTJ_list) - 1):
                    denom += diag_JTJ_list[1 + i]

                for i in range(len(diag_JTJ_list)):
                    diag_JTJ_list[i] /= denom
                    x_list[i] *= diag_JTJ_list[i]

                x = x_list[0].clone()
                for i in range(len(x_list) - 1):
                    x += x_list[1 + i]
                del x_list
                del diag_JTJ_list
                del denom

            elif gsgn.average_pcg_mode == "mean":
                x = torch.stack(x_list)
                x = x.mean(dim=0)
            elif gsgn.average_pcg_mode == "max":
                x = torch.stack(x_list)
                x = x.max(dim=0).values
            else:
                raise NotImplementedError("average_pcg_mode", gsgn.average_pcg_mode)

    # extract update per-parameter
    with measure_time("extract_update", timing_dict):
        # extract the parameter update from the PCG linear fitting
        x_dict = GaussianRasterizer.extract_gaussian_parameters(x, out_dict["data"])

        prev_xyz = gaussians._xyz.clone()
        prev_scaling = gaussians._scaling.clone()
        prev_rotation = gaussians._rotation.clone()
        prev_opacity = gaussians._opacity.clone()
        prev_features_dc = gaussians._features_dc.clone()
        prev_features_rest = gaussians.get_active_features_rest.clone()

    with measure_time("line_search", timing_dict):
        # line search
        if gsgn.line_search_use_maximum_gamma and gsgn.line_search_maximum_gamma > 0:
            if (iteration % gsgn.line_search_gamma_reset_interval) == 0:
                gamma = min(gsgn.line_search_initial_gamma, gsgn.line_search_maximum_gamma * 5)
            else:
                gamma = gsgn.line_search_maximum_gamma
        else:
            gamma = gsgn.line_search_initial_gamma

        alpha = gsgn.line_search_alpha
        prev_error = 1e15

        def update_params(g):
            gaussians._xyz.copy_((prev_xyz - g * gsgn.scale_fac_xyz * x_dict["xyz"]).float())
            gaussians._scaling.copy_((prev_scaling - g * gsgn.scale_fac_scale * x_dict["scaling"]).float())
            gaussians._rotation.copy_((prev_rotation - g * gsgn.scale_fac_rotation * x_dict["rotation"]).float())
            gaussians._opacity.copy_((prev_opacity - g * gsgn.scale_fac_opacity * x_dict["opacity"]).float())
            gaussians._features_dc.copy_((prev_features_dc - g * gsgn.scale_fac_features_dc * x_dict["features_dc"]).float())
            gaussians.get_active_features_rest.copy_((prev_features_rest - g * gsgn.scale_fac_features_rest * x_dict["features_rest"]).float())

        # select which images are used for line search
        line_search_images = [i for i in lm_step_images]
        random.shuffle(line_search_images)
        num_line_search_images = max(1, int(len(line_search_images) * gsgn.perc_images_in_line_search))
        line_search_images = line_search_images[:num_line_search_images]

        while True:
            update_params(gamma)

            # evaluate
            loss = 0.5 * render_and_add_to_residual_norm(
                gaussians=gaussians,
                viewpoint_stack=line_search_images,
                opt=opt,
                pipe=pipe,
                background=background
            )

            # backtracking cancel criterion
            if loss > prev_error:
                gamma /= alpha  # reverse last step
                # print('use gamma', gamma, 'with loss', prev_error, torch.mean(x), torch.median(x), torch.min(x.abs()), torch.max(x.abs()))
                break

            if gamma < 1e-10:
                # print('line-search terminated, gamma too low:', gamma, loss, torch.mean(x), torch.median(x), torch.min(x), torch.max(x))
                break

            # next backtracking step
            gamma = alpha * gamma
            prev_error = loss
        del line_search_images

        # keep track of maximum gamma. For example if gamma always in the range 0.01-0.1, then does not need to start searching from initial gamma all the time
        gsgn.line_search_maximum_gamma = max(gamma, gsgn.line_search_maximum_gamma)

    with measure_time("lm_update", timing_dict):
        # check LM trust_region_radius + apply or revert the update with the final gamma
        delta_x = - gamma * x  # final update vector, aka delta_x

        if gsgn.image_subsample_n_iters <= 1:
            # if not subsampling images or don't need sparse_jacobian --> use all images
            forward_output_for_cost_norm = forward_output
            images_for_cost_norm = lm_step_images
        else:
            # if we want to use apply_j for faster runtime + we subsample images --> we only have the last sparse_jacobian saved
            # therefore also only calculate the cost norms on the last forward_output (that corresponds to the sparse jacobians)
            forward_output_for_cost_norm = last_batch_forward_output
            images_for_cost_norm = last_batch_images

        with measure_time("residuals_to_gpu", timing_dict):
            forward_output_for_cost_norm.residuals = forward_output_for_cost_norm.residuals.to(delta_x.device, non_blocking=False)  # True
            if gsgn.compute_ssim_weights:
                forward_output_for_cost_norm.residuals_ssim = forward_output_for_cost_norm.residuals_ssim.to(delta_x.device, non_blocking=False)

        with measure_time("F_prev", timing_dict):
            F_prev = get_residual_norm(forward_output_for_cost_norm.residuals)
            if gsgn.compute_ssim_weights:
                F_prev += get_residual_norm(forward_output_for_cost_norm.residuals_ssim)

        with measure_time("F_Jx", timing_dict):
            with measure_time("F_Jx_reset_params", timing_dict):
                update_params(0)  # go back to old state to calculate F_Jx
            with measure_time("F_Jx_apply_j", timing_dict):
                x_resorted = gaussians.get_resorted_vec(delta_x)
                jx = apply_j(
                    x=delta_x,
                    x_resorted=x_resorted,
                    sparse_jacobians=out_dict["sparse_jacobians"],
                    index_map=out_dict["index_maps"],
                    per_gaussian_caches=out_dict["per_gaussian_caches"],
                    data=out_dict["data"],
                    timing_dict=timing_dict,
                    segments=out_dict["segments"],
                    segments_to_gaussians_list=out_dict["segments_to_gaussians"],
                    num_gaussians_in_block=out_dict["num_gaussians_in_block"],
                    block_offset_in_segments=out_dict["block_offset_in_segments"])
                del x_resorted

            with measure_time("F_Jx_clean_up", timing_dict):
                out_dict.pop("sparse_jacobians")
                out_dict.pop("index_maps")
                out_dict.pop("per_gaussian_caches")
                out_dict.pop("segments")
                out_dict.pop("segments_to_gaussians")
                out_dict.pop("num_gaussians_in_block")
                out_dict.pop("block_offset_in_segments")
                del delta_x
                # torch.cuda.empty_cache()

            with measure_time("F_Jx_compute", timing_dict):
                def compute_fjx(residuals: torch.Tensor, jx: torch.Tensor, weights: torch.Tensor = None):
                    # perform inplace calculations that do not require new tensors being created in memory
                    # can do this since weights / residuals are no longer required after this
                    if weights is not None:
                        weights *= jx
                        residuals += weights
                    else:
                        residuals += jx
                    residuals *= residuals
                    return 0.5 * residuals.sum()
                F_Jx = compute_fjx(forward_output_for_cost_norm.residuals, jx, out_dict["data"].weights if gsgn.compute_huber_weights else None)
                if gsgn.compute_ssim_weights:
                    F_Jx += compute_fjx(forward_output_for_cost_norm.residuals_ssim, jx, out_dict["data"].weights_ssim)
                del jx
                out_dict.pop("data")
                # torch.cuda.empty_cache()
            with measure_time("F_Jx_update_params", timing_dict):
                update_params(gamma)  # apply update with final gamma from line search

        with measure_time("F_new", timing_dict):
            # torch.cuda.empty_cache()

            if not use_all_images_at_once:
                # only render images to get new cost norm --> do it cheaper and don't store forward_output
                F_new = render_and_add_to_residual_norm(
                    gaussians=gaussians,
                    viewpoint_stack=images_for_cost_norm,
                    opt=opt,
                    pipe=pipe,
                    background=background,
                    compute_huber_weights=gsgn.compute_huber_weights,
                    huber_c=gsgn.huber_c,
                    compute_ssim_residuals=gsgn.compute_ssim_weights,
                    ssim_residual_scale_factor=gsgn.ssim_residual_scale_factor
                )
            else:
                forward_output = render_all_images_and_backward(
                    gaussians=gaussians,
                    viewpoint_stack=images_for_cost_norm,
                    opt=opt,
                    pipe=pipe,
                    background=background,
                    prepare_for_gsgn_backward=True,
                    compute_huber_weights=gsgn.compute_huber_weights,
                    huber_c=gsgn.huber_c,
                    compute_ssim_weights=gsgn.compute_ssim_weights,
                    ssim_residual_scale_factor=gsgn.ssim_residual_scale_factor
                )
                F_new = get_residual_norm(forward_output.residuals)
                if gsgn.compute_ssim_weights:
                    F_new += get_residual_norm(forward_output.residuals_ssim)

            del images_for_cost_norm

        model_cost_change = F_prev - F_Jx
        cost_change = F_prev - F_new

        # See CERES's TrustRegionStepEvaluator::StepAccepted() for a more complicated version of this
        terminate = False
        relative_decrease = cost_change / model_cost_change
        success = cost_change > 0 and relative_decrease > gsgn.min_relative_decrease
        if success:
            absolute_function_tolerance = F_prev * gsgn.function_tolerance
            if cost_change <= absolute_function_tolerance:
                terminate = True

            step_quality = relative_decrease
            min_factor = 1.0 / 3.0
            tmp_factor = 1.0 - (2 * step_quality - 1) ** 3
            trust_region_radius = trust_region_radius / max(min_factor, tmp_factor)
            trust_region_radius = min(trust_region_radius, gsgn.max_trust_region_radius)
            trust_region_radius = max(trust_region_radius, gsgn.min_trust_region_radius)
            radius_decrease_factor = 2.0
        else:
            print("no success, cost_change", cost_change, " model_cost_change", model_cost_change)
            trust_region_radius = trust_region_radius / radius_decrease_factor
            if cost_change < 0:
                print("reset to before iteration")
                # reverse update
                update_params(0)
                radius_decrease_factor = 2 * radius_decrease_factor

            if trust_region_radius <= gsgn.min_trust_region_radius:
                trust_region_radius = gsgn.min_trust_region_radius
                radius_decrease_factor = 2
                terminate = True

        del prev_xyz
        del prev_opacity
        del prev_scaling
        del prev_rotation
        del prev_features_dc
        del prev_features_rest
        del x
        del x_dict

    log_info = out_dict["log_info"]
    log_info["terminate"] = terminate
    log_info["success"] = success
    log_info["cost_change"] = cost_change
    log_info["model_cost_change"] = model_cost_change
    log_info["terminate"] = terminate
    log_info["gamma"] = gamma
    out_dict["n_used_images"] = len(lm_step_images)
    out_dict["loss"] = F_new
    out_dict["trust_region_radius"] = trust_region_radius
    out_dict["radius_decrease_factor"] = radius_decrease_factor
    out_dict["forward_output"] = forward_output

    return out_dict


def training(dataset: ModelParams, opt: OptimizationParams, gsgn: GSGNParams, pipe: PipelineParams, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, quiet: bool = False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt, gsgn, pipe, quiet)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    trust_region_radius = gsgn.trust_region_radius
    radius_decrease_factor = gsgn.radius_decrease_factor
    forward_output = None
    if checkpoint:
        print("load checkpoint from", checkpoint)
        try:
            (model_params, first_iter, loaded_trust_region_radius, loaded_radius_decrease_factor) = torch.load(checkpoint)
            if "before_gn" in checkpoint:
                first_iter = 0
        except:
            # is loading from pretrained model of 3DGS/distwar/3dgs_accel --> definitely set first_iter=0
            model_params, first_iter = torch.load(checkpoint)
            first_iter = 0

        # do not override trust_region_radius and radius_decrease_factor
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    timing_dict = {}

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_stack_indices = [i for i in range(len(viewpoint_stack))]
    if gsgn.image_subsample_frame_selection_mode == "random":
        random.shuffle(viewpoint_stack_indices)
    viewpoint_stack_sgd = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    tic = time.time()

    total_elapsed_time = 0

    @torch.enable_grad()
    def sgd_step(viewpoint_stack, iteration: int = -1):
        if iteration > -1:
            # update learning rate
            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if opt.increase_SH_deg_every > 0 and (iteration % opt.increase_SH_deg_every) == 0:
                gaussians.oneupSHdegree()

        # Pick a random camera
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = opt.lambda_l1 * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if opt.lambda_l2 > 0:
            Ll2 = l2_loss(image, gt_image)
            loss = loss + opt.lambda_l2 * Ll2
        loss.backward()

        # Densification
        with torch.no_grad():
            if iteration > -1 and iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    n_points_before_densification, n_points_after_densification, n_points_after_prune = gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalars("densification", {
                            "before_densification": n_points_before_densification,
                            "after_densification": n_points_after_densification,
                            "after_prune": n_points_after_prune
                        }, iteration)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

        # Optimizer step
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

    for iteration in range(first_iter, opt.iterations + 1):
        if not quiet:
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

        with measure_time("iter", timing_dict):
            # sgd initialization
            if gsgn.num_sgd_iterations_before_gn > 0 and iteration == 1:
                for i in tqdm(range(gsgn.num_sgd_iterations_before_gn), desc="sgd_before_gsgn", leave=True):
                    with measure_time("sgd_before_gsgn", timing_dict, additive=True):
                        if not viewpoint_stack_sgd:
                            viewpoint_stack_sgd = scene.getTrainCameras().copy()
                        sgd_step(viewpoint_stack_sgd, 1 + i)
                toc = time.time()
                total_elapsed_time += toc - tic
                tic = time.time()

                _iteration = 0 # hacky iteration identifier for the SGD initialization
                if not quiet:
                    log_render_stats(
                        tb_writer=tb_writer,
                        iteration=_iteration,
                        testing_iterations=testing_iterations,
                        scene=scene,
                        renderFunc=render,
                        renderArgs=(pipe, background)
                    )

                if _iteration in checkpoint_iterations:                 
                    mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                    stats = {
                        "mem": mem,
                        "ellipse_time": total_elapsed_time,
                        "num_GS": len(gaussians.get_xyz),
                    }
                    with open(scene.model_path + "/train_stats_" + str(_iteration) + ".json", "w") as f:
                        json.dump(stats, f)
                    print("\n[ITER {}] Saving Gaussians (after SGD)".format(_iteration))
                    scene.save(_iteration)

                if _iteration in checkpoint_iterations:
                    print("\n[ITER {}] Saving Checkpoint (after SGD)".format(_iteration))
                    torch.save((gaussians.capture(), _iteration, None, None), scene.model_path + "/chkpnt" + str(_iteration) + ".pth")

            # free SGD specific memory that is no longer used
            if gsgn.num_sgd_iterations_between_gn <= 0 and gsgn.num_sgd_iterations_after_gn <= 0 and len(checkpoint_iterations) == 0:
                gaussians.remove_sgd_data()

            # compute LM update
            out_dict = lm_step(
                gaussians=gaussians,
                viewpoint_stack=viewpoint_stack,
                viewpoint_stack_indices=viewpoint_stack_indices,
                opt=opt,
                pipe=pipe,
                background=background,
                gsgn=gsgn,
                iteration=iteration,
                trust_region_radius=trust_region_radius,
                radius_decrease_factor=radius_decrease_factor,
                timing_dict=timing_dict,
                forward_output=forward_output
            )
            trust_region_radius = out_dict["trust_region_radius"]
            radius_decrease_factor = out_dict["radius_decrease_factor"]
            forward_output = out_dict["forward_output"]

            # do sgd iters between LM
            if gsgn.num_sgd_iterations_between_gn > 0 and (iteration % gsgn.sgd_between_gn_every) == 0:
                # gaussians.reset_optimizer()
                for i in tqdm(range(gsgn.num_sgd_iterations_between_gn), desc="sgd_between_gsgn", leave=True):
                    with measure_time("sgd_between_gsgn", timing_dict, additive=True):
                        if not viewpoint_stack_sgd:
                            viewpoint_stack_sgd = scene.getTrainCameras().copy()
                        # sgd_step(viewpoint_stack_sgd, 1 + previous_sgd_iters + i)
                        sgd_step(viewpoint_stack_sgd, -1)

            # do sgd iters after LM
            if gsgn.num_sgd_iterations_after_gn > 0 and iteration == opt.iterations:
                # gaussians.reset_optimizer()
                # previous_sgd_iters = gsgn.num_sgd_iterations_before_gn + (iteration // gsgn.sgd_between_gn_every) * gsgn.num_sgd_iterations_between_gn
                for i in tqdm(range(gsgn.num_sgd_iterations_after_gn), desc="sgd_after_gsgn", leave=True):
                    with measure_time("sgd_after_gsgn", timing_dict, additive=True):
                        if not viewpoint_stack_sgd:
                            viewpoint_stack_sgd = scene.getTrainCameras().copy()
                        # sgd_step(viewpoint_stack_sgd, 1 + previous_sgd_iters + i)
                        sgd_step(viewpoint_stack_sgd, -1)

        toc = time.time()
        total_elapsed_time += toc - tic

        # post-iteration updates
        with torch.no_grad():
            # Progress bar
            progress_bar.set_postfix({
                "Loss": f"{out_dict['loss']:.{7}f}",
                "Max-Mem": f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}"
            })
            progress_bar.update(1)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if not quiet:
                training_report(tb_writer, iteration, out_dict, timing_dict, testing_iterations, scene, render, (pipe, background), gaussians.total_params, gaussians.num_gaussians, viewpoint_stack[0].image_width, viewpoint_stack[0].image_height)
            if iteration in saving_iterations:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                stats = {
                    "mem": mem,
                    "ellipse_time": total_elapsed_time,
                    "num_GS": len(gaussians.get_xyz),
                }
                with open(scene.model_path + "/train_stats_" + str(iteration) + ".json", "w") as f:
                    json.dump(stats, f)
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint (after LM)".format(iteration))
                torch.save((gaussians.capture(), iteration, trust_region_radius, radius_decrease_factor), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        tic = time.time()

    return scene


def prepare_output_and_logger(model: ModelParams, opt: OptimizationParams, gn: GSGNParams, pipe: PipelineParams, quiet: bool = False):
    if not model.model_path:
        unique_str = str(uuid.uuid4())
        exp_name = f"{model.exp_name}_" if model.exp_name is not None else ""
        model.model_path = os.path.join(model.root_out, f"{exp_name}{unique_str[0:10]}")
        
    # Set up output folder
    print("Output folder: {}".format(model.model_path))
    os.makedirs(model.model_path, exist_ok=True)
    with open(os.path.join(model.model_path, "cfg_args.json"), 'w') as cfg_log_f:
        all_args = {
            "model": {**vars(model)},
            "opt": {**vars(opt)},
            "gn": {**vars(gn)},
            "pipe": {**vars(pipe)}
        }
        json.dump(all_args, cfg_log_f, indent=4)
    with open(os.path.join(model.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if not quiet:
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter(model.model_path)
        else:
            print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def log_render_stats(tb_writer, iteration, testing_iterations, scene : Scene, renderFunc, renderArgs, ):
    torch.cuda.empty_cache()
    # validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
    #                       {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

    validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                          {'name': 'train', 'cameras': scene.getTrainCameras()})

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            l2_test = 0.0
            psnr_test = 0.0
            for idx, viewpoint in enumerate(config['cameras']):
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if tb_writer and (idx < 5):
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                l1_test += l1_loss(image, gt_image).mean().double().item()
                l2_test += l2_loss(image, gt_image).mean().double().item()
                psnr_test += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double().item()
            psnr_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            l2_test /= len(config['cameras'])
            print("\n[ITER {}] Evaluating {}: L1 {} L2 {} PSNR {}".format(iteration, config['name'], l1_test, l2_test, psnr_test))
            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l2_loss', l2_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)


def training_report(tb_writer, iteration, out_dict, timing_dict, testing_iterations, scene : Scene, renderFunc, renderArgs, num_params, num_gaussians, image_width, image_height):
    if tb_writer:
        tb_writer.add_scalar('train/loss', out_dict["loss"].item(), iteration)

        tb_writer.add_scalars(f'time', timing_dict, iteration)
        timing_dict.clear()

        tb_writer.add_scalar('sparse_j/total_size_in_gb', out_dict["log_info"]["total_size_in_gb"], iteration)
        tb_writer.add_scalar('sparse_j/mean_n_contrib_per_pixel', out_dict["log_info"]["mean_n_contrib_per_pixel"], iteration)
        tb_writer.add_scalar('sparse_j/num_params', num_params, iteration)
        tb_writer.add_scalar('sparse_j/num_gaussians', num_gaussians, iteration)
        tb_writer.add_scalar('sparse_j/image_width', image_width, iteration)
        tb_writer.add_scalar('sparse_j/image_height', image_height, iteration)

        tb_writer.add_scalar('lm/trust_region_radius', out_dict["trust_region_radius"].item() if isinstance(out_dict["trust_region_radius"], torch.Tensor) else out_dict["trust_region_radius"], iteration)
        tb_writer.add_scalar('lm/radius_decrease_factor', out_dict["radius_decrease_factor"], iteration)
        tb_writer.add_scalar('lm/terminate', out_dict["log_info"]["terminate"], iteration)
        tb_writer.add_scalar('lm/success', out_dict["log_info"]["success"], iteration)
        tb_writer.add_scalar('lm/cost_change', out_dict["log_info"]["cost_change"].item(), iteration)
        tb_writer.add_scalar('lm/model_cost_change', out_dict["log_info"]["model_cost_change"].item(), iteration)
        tb_writer.add_scalar('lm/line_search_gamma', out_dict["log_info"]["gamma"], iteration)

        x = out_dict["x"]
        xf = x[x.abs() > 0]
        if xf.numel() > 0:
            tb_writer.add_scalar('grad/min', xf.abs().min().item(), iteration)
        tb_writer.add_scalar('grad/max', x.abs().max().item(), iteration)
        tb_writer.add_scalar('grad/mean', x.mean().item(), iteration)
        tb_writer.add_scalar('grad/std', x.std().item(), iteration)

        cg_info = out_dict["log_info"]["pcg_info"]
        if cg_info is not None:
            tb_writer.add_scalar('pcg/optimal', cg_info["optimal"], iteration)
            tb_writer.add_scalar('pcg/niter', cg_info["niter"], iteration)
            tb_writer.add_scalar('pcg/err', cg_info["err"], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        log_render_stats(
            tb_writer=tb_writer,
            iteration=iteration,
            testing_iterations=testing_iterations,
            scene=scene,
            renderFunc=renderFunc,
            renderArgs=renderArgs
        )

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    gn = GSGNParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--render_and_eval_train", action='store_true', default=False)
    parser.add_argument("--render_and_eval_spherical", action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    old_std_out = safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset_args = lp.extract(args)
    scene = training(dataset_args, op.extract(args), gn.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.quiet)

    # All done
    print("\nTraining complete.")
    safe_state(False, old_std_out)  # allow print statements for eval

    loaded_cams = {
        "train": scene.train_cameras,
        "test": scene.test_cameras,
        "spherical": scene.spherical_cameras,
        "extent": scene.cameras_extent,
    }
    for iter in args.save_iterations:
        render_sets(
            dataset_args,
            iter,
            pp.extract(args),
            skip_test=False,
            skip_train=not args.render_and_eval_train,
            skip_spherical=not args.render_and_eval_spherical,
            loaded_cams=loaded_cams
        )
    evaluate([dataset_args.model_path])
