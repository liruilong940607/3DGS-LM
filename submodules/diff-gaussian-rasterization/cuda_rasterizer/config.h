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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define GSGN_NUM_CHANNELS 3 // Default 3, RGB
#define GSGN_BLOCK_X 16
#define GSGN_BLOCK_Y 16
#define GSGN_ALPHA_THRESH 1.0f / 255.0f

#endif