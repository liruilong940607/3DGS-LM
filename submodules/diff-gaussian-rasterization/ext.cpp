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

#include <torch/extension.h>
#include "rasterize_points.h"
#include "cuda_rasterizer/gsgn_data_spec.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
    m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
    m.def("mark_visible", &markVisible);

    // -----------------------------------
    // ADDITIONAL METHODS FOR GN SUPPORT |
    // -----------------------------------
    m.def("eval_jtf_and_get_sparse_jacobian", &EvalJTFAndGetSparseJacobian);
    m.def("calc_preconditioner", &CalcPreconditioner);
    m.def("apply_jtj", &ApplyJTJ);    
    m.def("apply_j", &ApplyJ);
    m.def("sort_sparse_jacobians", &SortSparseJacobians);
    m.def("filter_reordered_geometry_buffer", &FilterReorderedGeometryBuffer);

    py::class_<GSGNDataSpec>(m, "GSGNDataSpec")
        .def(py::init<>())
        .def_readwrite("background", &GSGNDataSpec::background)
        .def_readwrite("params", &GSGNDataSpec::params)
        .def_readwrite("means3D", &GSGNDataSpec::means3D)
        .def_readwrite("scale_modifier", &GSGNDataSpec::scale_modifier)
        .def_readwrite("viewmatrix", &GSGNDataSpec::viewmatrix)
        .def_readwrite("projmatrix", &GSGNDataSpec::projmatrix)
        .def_readwrite("tan_fovx", &GSGNDataSpec::tan_fovx)
        .def_readwrite("tan_fovy", &GSGNDataSpec::tan_fovy)
        .def_readwrite("cx", &GSGNDataSpec::cx)
        .def_readwrite("cy", &GSGNDataSpec::cy)
        .def_readwrite("sh", &GSGNDataSpec::sh)
        .def_readwrite("unactivated_opacity", &GSGNDataSpec::unactivated_opacity)
        .def_readwrite("unactivated_scales", &GSGNDataSpec::unactivated_scales)
        .def_readwrite("unactivated_rotations", &GSGNDataSpec::unactivated_rotations)
        .def_readwrite("degree", &GSGNDataSpec::degree)
        .def_readwrite("campos", &GSGNDataSpec::campos)
        .def_readwrite("geomBuffer", &GSGNDataSpec::geomBuffer)
        .def_readwrite("R", &GSGNDataSpec::R)
        .def_readwrite("binningBuffer", &GSGNDataSpec::binningBuffer)
        .def_readwrite("imageBuffer", &GSGNDataSpec::imageBuffer)
        .def_readwrite("debug", &GSGNDataSpec::debug)
        .def_readwrite("use_double_precision", &GSGNDataSpec::use_double_precision)
        .def_readwrite("n_contrib_vol_rend_prefix_sum", &GSGNDataSpec::n_contrib_vol_rend_prefix_sum)
        .def_readwrite("residuals", &GSGNDataSpec::residuals)        
        .def_readwrite("weights", &GSGNDataSpec::weights)
        .def_readwrite("residuals_ssim", &GSGNDataSpec::residuals_ssim)
        .def_readwrite("weights_ssim", &GSGNDataSpec::weights_ssim)
        .def_readwrite("map_visible_gaussians", &GSGNDataSpec::map_visible_gaussians)
        .def_readwrite("map_cache_to_gaussians", &GSGNDataSpec::map_cache_to_gaussians)
        .def_readwrite("num_visible_gaussians", &GSGNDataSpec::num_visible_gaussians)
        .def_readwrite("num_sparse_gaussians", &GSGNDataSpec::num_sparse_gaussians)
        .def_readwrite("P", &GSGNDataSpec::P)
        .def_readwrite("M", &GSGNDataSpec::M)
        .def_readwrite("H", &GSGNDataSpec::H)
        .def_readwrite("W", &GSGNDataSpec::W)
        .def_readwrite("num_images", &GSGNDataSpec::num_images)
        .def_readwrite("jx_stride", &GSGNDataSpec::jx_stride)
        .def_readwrite("offset_xyz", &GSGNDataSpec::offset_xyz)
        .def_readwrite("offset_scales", &GSGNDataSpec::offset_scales)
        .def_readwrite("offset_rotations", &GSGNDataSpec::offset_rotations)
        .def_readwrite("offset_opacity", &GSGNDataSpec::offset_opacity)
        .def_readwrite("offset_features_dc", &GSGNDataSpec::offset_features_dc)
        .def_readwrite("offset_features_rest", &GSGNDataSpec::offset_features_rest)
        .def_readwrite("total_params", &GSGNDataSpec::total_params);
}