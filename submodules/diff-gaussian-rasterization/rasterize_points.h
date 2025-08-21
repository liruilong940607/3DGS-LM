/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <vector>
#include <string>
#include "cuda_rasterizer/gsgn_data_spec.h"

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const float cx,
    const float cy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
    const bool prepare_for_gsgn_backward,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const float cx,
    const float cy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);

torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix);

// -----------------------------------
// ADDITIONAL METHODS FOR GN SUPPORT |
// -----------------------------------

std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> EvalJTFAndGetSparseJacobian(GSGNDataSpec& data);
torch::Tensor ApplyJTJ(GSGNDataSpec& data, torch::Tensor x_vec, torch::Tensor x_resorted_vec, std::vector<torch::Tensor> sparse_jacobians, std::vector<torch::Tensor> index_map, std::vector<torch::Tensor> per_gaussian_cache, std::vector<torch::Tensor> segments, std::vector<torch::Tensor> segments_to_gaussians, std::vector<torch::Tensor> num_gaussians_in_block, std::vector<torch::Tensor> block_offset_in_segments);
torch::Tensor CalcPreconditioner(GSGNDataSpec& data, std::vector<torch::Tensor> sparse_jacobians, std::vector<torch::Tensor> index_map, std::vector<torch::Tensor> per_gaussian_cache, std::vector<torch::Tensor> segments, std::vector<torch::Tensor> segments_to_gaussians, std::vector<torch::Tensor> num_gaussians_in_block, std::vector<torch::Tensor> block_offset_in_segments);
torch::Tensor ApplyJ(GSGNDataSpec& data, torch::Tensor x_vec, torch::Tensor x_resorted_vec, std::vector<torch::Tensor> sparse_jacobians, std::vector<torch::Tensor> index_map, std::vector<torch::Tensor> per_gaussian_cache, std::vector<torch::Tensor> segments, std::vector<torch::Tensor> segments_to_gaussians, std::vector<torch::Tensor> num_gaussians_in_block, std::vector<torch::Tensor> block_offset_in_segments);
std::vector<torch::Tensor> SortSparseJacobians(GSGNDataSpec& data, std::vector<torch::Tensor> sparse_jacobians, std::vector<torch::Tensor> indices);
std::vector<torch::Tensor> FilterReorderedGeometryBuffer(std::vector<torch::Tensor> geomBuffer, std::vector<torch::Tensor> map_cache_to_gaussians, std::vector<int> num_visible_gaussians);
