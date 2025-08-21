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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <tuple>
#include <functional>
#include "gsgn_data_spec.h"

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

        static std::tuple<int, int64_t> forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
            const float cx, float cy,
			const bool prefiltered,
			float* out_color,
            int* n_contrib_vol_rend,
            bool* is_gaussian_hit,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
            const float cx, float cy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool debug);

        // -----------------------------------
        // ADDITIONAL METHODS FOR GN SUPPORT |
        // -----------------------------------

        static void reorder_geometry_buffer(const int P, char* geom_buffer, int* radii, std::function<char* (size_t)> out_geom_buffer);

        static void filter_reordered_geometry_buffer(const int num_visible_gaussians, int* map_cache_to_gaussians, char* geom_buffer, std::function<char* (size_t)> out_geom_buffer);

        template<typename T> static void eval_jtf_and_get_sparse_jacobian(PackedGSGNDataSpec data, T* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache);

        template<typename T> static void apply_j(PackedGSGNDataSpec data, T* x_vec, T* x_resorted_vec, T* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr);

        template<typename T> static void apply_jt(PackedGSGNDataSpec data, T* g_vec, T* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr);

        template<typename T> static void calc_preconditioner(PackedGSGNDataSpec data, T* M_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block);

        template<typename T> static void sort_sparse_jacobians(PackedGSGNDataSpec data, T** in_sparse_jacobians, T** out_sparse_jacobians, int64_t** indices);
	};
};

#endif