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

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

    struct GeometryStateReduced
	{
        // reduced struct that only contains the values needed for backward()
        // we convert float2 and float4 into float arrays, because this reduces the size of the struct
        // for some reason sizeof(this) = 80 with float2/float4, but sizeof(this) = 64 with float arrays
        float means2D[2];
        float conic_opacity[4];
        float cov3D[6];
        float rgb[3];
        bool clamped[3];
		bool radius_gt_zero;
	};

    struct GaussianAttributeNoSH
    {
        float mean3D[3];
        float unactivated_rotation[4];
        float unactivated_scale[3];
        float unactivated_opacity;
    };

    struct GaussianCache
    {
        // 23 floats for computeCov2D()
        float t[4];
        float T[6];
        float a_;
        float b_;
        float c_;
        float x_grad_mul;
        float y_grad_mul;
        float dL_dT_precomp[6];
        float denom;
        float denom2inv;
        // 18 floats for preprocess()
        float R[9];
        float dRGBdx[3];
        float dRGBdy[3];
        float dRGBdz[3];
    };

    struct GaussianCacheComputeCov2D
    {
        // 23 floats for computeCov2D()
        float t[4];
        float T[6];
        float a_;
        float b_;
        float c_;
        float x_grad_mul;
        float y_grad_mul;
        float dL_dT_precomp[6];
        float denom;
        float denom2inv;
    };

    struct GaussianCachePreprocess
    {
        // 18 floats for preprocess()
        float R[9];
        float dRGBdx[3];
        float dRGBdy[3];
        float dRGBdz[3];
    };

    // struct GaussianCache
    // {
    //     GaussianCacheComputeCov2D cov2D;
    //     GaussianCachePreprocess preprocess;
    // };

    struct __align__(8) GradientCache
    {
        __half dchannel_dcolor;
        __half dL_dalpha[3];
    };

	struct ImageState
	{
		uint2* ranges;
		int32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
        uint32_t* point_list;
		uint32_t* point_list_unsorted;
        uint64_t* point_list_keys;
        uint64_t* point_list_keys_unsorted;
        size_t sorting_size;
        char* list_sorting_space;
		
		static BinningState fromChunk(char*& chunk, size_t P);
	};

    struct BinningStateReduced
	{
		uint32_t* point_list;

		static BinningStateReduced fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};