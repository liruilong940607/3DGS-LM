#pragma once

#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include "config.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

using torch::Tensor;

#define TORCH_CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define TORCH_CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) TORCH_CHECK_CUDA(x); TORCH_CHECK_CONTIGUOUS(x)

struct GSGNDataSpec {
    // inputs provided all the time
    Tensor background;
    Tensor params;
	Tensor means3D;
	float scale_modifier;
	std::vector<Tensor> viewmatrix;
    std::vector<Tensor> projmatrix;
	Tensor tan_fovx;
	Tensor tan_fovy;
    Tensor cx;
    Tensor cy;
	Tensor sh;
    Tensor unactivated_opacity;
	Tensor unactivated_scales;
	Tensor unactivated_rotations;
	int degree; // degree of SH coeffs (0, 1, 2, 3)
    int H; // image height
    int W; // image width
	Tensor campos;
	std::vector<Tensor> geomBuffer;
	std::vector<int> R;  // num_rendered_list
	std::vector<Tensor> binningBuffer;
	std::vector<Tensor> imageBuffer;
    std::vector<int> num_sparse_gaussians;
    std::vector<Tensor> map_visible_gaussians;
    std::vector<Tensor> map_cache_to_gaussians;
    std::vector<int> num_visible_gaussians;
	bool debug;
    bool use_double_precision;

    // inputs that can be optional (not used by every GSGN function)
    bool have_n_contrib_vol_rend_prefix_sum = false;
    std::vector<Tensor> n_contrib_vol_rend_prefix_sum;
    bool have_residuals = false;
    Tensor residuals;
    bool have_weights = false;
    Tensor weights;
    bool have_residuals_ssim = false;
    Tensor residuals_ssim;
    bool have_weights_ssim = false;
    Tensor weights_ssim;

    // pointers created for later usage
    int* num_rendered_ptrs;
    float** viewmatrix_ptrs;
    float** projmatrix_ptrs;
    char** geomBuffer_ptrs;
    char** binningBuffer_ptrs;
    char** imageBuffer_ptrs;
    int** n_contrib_vol_rend_prefix_sum_ptrs;
    int* n_sparse_gaussians;
    int32_t max_n_sparse_gaussians = 0;
    int** map_visible_gaussians_ptrs;
    int** map_cache_to_gaussians_ptrs;
    int* n_visible_gaussians;
    int32_t max_n_visible_gaussians = 0;

    // extracted values
    int P; // number of gaussians
    uint32_t num_images; // number of images
    int num_pixels; // number of pixels in each image
    int jx_stride;
    int M = 0; // total number of SH coeffs per channel (1, 4, 9, 16)
    Tensor focal_x, focal_y; // calculated from tan_fovx and tan_fovy

    // param offset values for input/output vectors
    int64_t total_params;
    int64_t offset_xyz, offset_scales, offset_rotations, offset_opacity, offset_features_dc, offset_features_rest;

    inline void check() {
        CHECK_INPUT(background);
        CHECK_INPUT(params);
        CHECK_INPUT(means3D);
        CHECK_INPUT(tan_fovx);
        CHECK_INPUT(tan_fovy);
        CHECK_INPUT(cx);
        CHECK_INPUT(cy);
        CHECK_INPUT(sh);
        CHECK_INPUT(unactivated_opacity);
        CHECK_INPUT(unactivated_scales);
        CHECK_INPUT(unactivated_rotations);
        CHECK_INPUT(campos);

        // extract values
        assert(means3D.size(0) == params.size(0));
        P = params.size(0);
        num_images = geomBuffer.size();
        num_pixels = H * W;
        focal_x = W / (2.0f * tan_fovx);
        focal_y = H / (2.0f * tan_fovy);
        assert(sh.size(0) != 0);
        M = sh.size(1) + 1; // sh only starts from second coeff, because sh[0] is not needed in backward passes

        jx_stride = num_images * W * H;
        // int alignment = 128;
        // jx_stride = (jx_stride + alignment - 1) & ~(alignment - 1);

        // assert that we actually have batch input
        have_n_contrib_vol_rend_prefix_sum = n_contrib_vol_rend_prefix_sum.size() == num_images;
        have_residuals = residuals.size(0) != 0;
        have_weights = weights.size(0) != 0;
        have_residuals_ssim = residuals_ssim.size(0) != 0;
        have_weights_ssim = weights_ssim.size(0) != 0;
        assert(viewmatrix.size() == num_images);
        assert(projmatrix.size() == num_images);
        assert(tan_fovx.size(0) == num_images);
        assert(tan_fovy.size(0) == num_images);
        assert(cx.size(0) == num_images);
        assert(cy.size(0) == num_images);
        assert(campos.size(0) == num_images);
        assert(geomBuffer.size() == num_images);
        assert(binningBuffer.size() == num_images);
        assert(imageBuffer.size() == num_images);
        assert(R.size() == num_images);
        assert(num_sparse_gaussians.size() == num_images);
        assert(num_visible_gaussians.size() == num_images);
        assert(map_visible_gaussians.size() == num_images);
        assert(map_cache_to_gaussians.size() == num_images);
        assert(n_contrib_vol_rend_prefix_sum.size() == num_images || !have_n_contrib_vol_rend_prefix_sum);

        if(have_residuals) {
            CHECK_INPUT(residuals);
        }
        if(have_weights) {
            CHECK_INPUT(weights);
        }
        if(have_residuals_ssim) {
            CHECK_INPUT(residuals_ssim);
        }
        if(have_weights_ssim) {
            CHECK_INPUT(weights_ssim);
        }

        for(uint32_t i=0; i < num_images; i++) {
            CHECK_INPUT(viewmatrix[i]);
            CHECK_INPUT(projmatrix[i]);
            CHECK_INPUT(geomBuffer[i]);
            CHECK_INPUT(binningBuffer[i]);
            CHECK_INPUT(imageBuffer[i]);
            CHECK_INPUT(map_visible_gaussians[i]);
            CHECK_INPUT(map_cache_to_gaussians[i]);
            if(have_n_contrib_vol_rend_prefix_sum) {
                CHECK_INPUT(n_contrib_vol_rend_prefix_sum[i]);
            }
        }
    }

    inline void allocate_pointer_memory() {
        cudaMallocManaged((void**) &num_rendered_ptrs, num_images * sizeof(int));
        cudaMallocManaged((void**) &viewmatrix_ptrs, num_images * sizeof(float*));
        cudaMallocManaged((void**) &projmatrix_ptrs, num_images * sizeof(float*));
        cudaMallocManaged((void**) &geomBuffer_ptrs, num_images * sizeof(char*));
        cudaMallocManaged((void**) &binningBuffer_ptrs, num_images * sizeof(char*));
        cudaMallocManaged((void**) &imageBuffer_ptrs, num_images * sizeof(char*));
        cudaMallocManaged((void**) &map_visible_gaussians_ptrs, num_images * sizeof(int*));
        cudaMallocManaged((void**) &map_cache_to_gaussians_ptrs, num_images * sizeof(int*));
        cudaMallocManaged((void**) &n_visible_gaussians, num_images * sizeof(int));
        cudaMallocManaged((void**) &n_sparse_gaussians, num_images * sizeof(int));
        if(have_n_contrib_vol_rend_prefix_sum) {
            cudaMallocManaged((void**) &n_contrib_vol_rend_prefix_sum_ptrs, num_images * sizeof(int*));
        }

        for(uint32_t i=0; i < num_images; i++) {
            num_rendered_ptrs[i] = R[i];
            viewmatrix_ptrs[i] = viewmatrix[i].data_ptr<float>();
            projmatrix_ptrs[i] = projmatrix[i].data_ptr<float>();
            geomBuffer_ptrs[i] = reinterpret_cast<char*>(geomBuffer[i].data_ptr());
            binningBuffer_ptrs[i] = reinterpret_cast<char*>(binningBuffer[i].data_ptr());
            imageBuffer_ptrs[i] = reinterpret_cast<char*>(imageBuffer[i].data_ptr());
            map_visible_gaussians_ptrs[i] = map_visible_gaussians[i].data_ptr<int>();
            map_cache_to_gaussians_ptrs[i] = map_cache_to_gaussians[i].data_ptr<int>();
            int x = num_visible_gaussians[i];
            n_visible_gaussians[i] = x;
            max_n_visible_gaussians = x > max_n_visible_gaussians ? x : max_n_visible_gaussians;
            x = num_sparse_gaussians[i];
            n_sparse_gaussians[i] = x;
            max_n_sparse_gaussians = x > max_n_sparse_gaussians ? x : max_n_sparse_gaussians;
            if(have_n_contrib_vol_rend_prefix_sum) {
                n_contrib_vol_rend_prefix_sum_ptrs[i] = n_contrib_vol_rend_prefix_sum[i].data_ptr<int>();
            }
        }
    }

    inline void free_pointer_memory() {
        cudaDeviceSynchronize();
        cudaFree(num_rendered_ptrs);
        cudaFree(viewmatrix_ptrs);
        cudaFree(projmatrix_ptrs);
        cudaFree(geomBuffer_ptrs);
        cudaFree(binningBuffer_ptrs);
        cudaFree(imageBuffer_ptrs);
        cudaFree(n_sparse_gaussians);
        cudaFree(map_visible_gaussians_ptrs);
        cudaFree(map_cache_to_gaussians_ptrs);
        cudaFree(n_visible_gaussians);
        if(have_n_contrib_vol_rend_prefix_sum) {
            cudaFree(n_contrib_vol_rend_prefix_sum_ptrs);
        }
    }

    inline void get_parameter_offsets() {
        // get params and their offset
        // we first write all xyz params of all gaussians, then all scales params, ...

        int64_t params_xyz = P * 3;
        offset_xyz = 0;

        int64_t params_scales = P * 3;
        offset_scales = offset_xyz + params_xyz;

        int64_t params_rotations = P * 4;
        offset_rotations = offset_scales + params_scales;

        int64_t params_opacity = P; // P * 1
        offset_opacity = offset_rotations + params_rotations;

        // sh has shape (P, M, 3) where M is the number of sh coeffs per rgb channel
        // following the tensor layout in python, we divide sh into:
        // _features_dc --> (P, 1, 3): the first sh coeff per rgb channel (e.g., diffuse color)
        // _features_rest --> (P, M-1, 3): the remaining sh choeffs per rgb channel (e.g., specular colors)
        int64_t params_features_dc = P * GSGN_NUM_CHANNELS; 
        offset_features_dc = offset_opacity + params_opacity;

        int32_t params_per_channel_rest = (M - 1);
        int64_t params_features_rest = P * params_per_channel_rest * GSGN_NUM_CHANNELS; 
        offset_features_rest = offset_features_dc + params_features_dc;

        total_params = params_xyz + params_scales + params_rotations + params_opacity + params_features_dc + params_features_rest;
    }

    inline void init() {
        check();
        allocate_pointer_memory();
        get_parameter_offsets();
    }
};

namespace CudaRasterizer
{
    struct PackedGSGNDataSpec {
        PackedGSGNDataSpec(GSGNDataSpec& data):
            P(data.P),
            D(data.degree),
            M(data.M),
            num_rendered(data.num_rendered_ptrs),
            background(data.background.data_ptr<float>()),
            W(data.W),
            H(data.H),
            num_pixels(data.num_pixels),
            jx_stride(data.jx_stride),
            num_images(data.num_images),
            params(data.params.data_ptr<float>()),
            means3D((float3*)data.means3D.data_ptr<float>()),
            shs(data.sh.data_ptr<float>()),
            unactivated_opacity(data.unactivated_opacity.data_ptr<float>()),
            unactivated_scales((float3*)data.unactivated_scales.data_ptr<float>()),
            unactivated_rotations((float4*)data.unactivated_rotations.data_ptr<float>()),
            scale_modifier(data.scale_modifier),
            viewmatrix(data.viewmatrix_ptrs),
            projmatrix(data.projmatrix_ptrs),
            campos((glm::vec3*)data.campos.data_ptr<float>()),
            tan_fovx(data.tan_fovx.data_ptr<float>()),
            tan_fovy(data.tan_fovy.data_ptr<float>()),
            cx(data.cx.data_ptr<float>()),
            cy(data.cy.data_ptr<float>()),
            focal_x(data.focal_x.data_ptr<float>()),
            focal_y(data.focal_y.data_ptr<float>()),
            have_n_contrib_vol_rend_prefix_sum(data.have_n_contrib_vol_rend_prefix_sum),
            n_contrib_vol_rend_prefix_sum(data.n_contrib_vol_rend_prefix_sum_ptrs),
            have_residuals(data.have_residuals),
            residuals(data.residuals.data_ptr<float>()),
            have_weights(data.have_weights),
            weights(data.weights.data_ptr<float>()),
            have_residuals_ssim(data.have_residuals_ssim),
            residuals_ssim(data.residuals_ssim.data_ptr<float>()),
            have_weights_ssim(data.have_weights_ssim),
            weights_ssim(data.weights_ssim.data_ptr<float>()),
            map_visible_gaussians(data.map_visible_gaussians_ptrs),
            map_cache_to_gaussians(data.map_cache_to_gaussians_ptrs),
            n_visible_gaussians(data.n_visible_gaussians),
            debug(data.debug),
            offset_xyz(data.offset_xyz),
            offset_scales(data.offset_scales),
            offset_rotations(data.offset_rotations),
            offset_opacity(data.offset_opacity),
            offset_features_dc(data.offset_features_dc),
            offset_features_rest(data.offset_features_rest),
            n_sparse_gaussians(data.n_sparse_gaussians) {

            max_n_sparse_gaussians = data.max_n_sparse_gaussians;
            max_n_visible_gaussians = data.max_n_visible_gaussians;

            // have to re-allocate them, because the pointers will get modified when obtaining the chunks, e.g. see implementation of allocate_pointer_memory()
            // if we call allocate_pointer_memory() from two different PackedGSGNDataSpec objects, that were constructed using the same GSGNDataSpec, it would thus not work the second time.
            cudaMallocManaged((void**) &geomBuffer_ptrs, num_images * sizeof(char*));
            cudaMallocManaged((void**) &binningBuffer_ptrs, num_images * sizeof(char*));
            cudaMallocManaged((void**) &imageBuffer_ptrs, num_images * sizeof(char*));
            for(int i = 0; i < num_images; i++) {
                geomBuffer_ptrs[i] = data.geomBuffer_ptrs[i];
                binningBuffer_ptrs[i] = data.binningBuffer_ptrs[i];
                imageBuffer_ptrs[i] = data.imageBuffer_ptrs[i];
            }
        }

        // inputs from GSGNDataSpec
        int P;
        int D;
        int M;
        int* num_rendered;
        float* __restrict__ background;
        int W;
        int H;
        int num_pixels;
        int jx_stride;
        int num_images;
        float* __restrict__ params;
        float3* __restrict__ means3D;
        float* __restrict__ shs;
        float* __restrict__ unactivated_opacity;
        float3* __restrict__ unactivated_scales;
        float4* __restrict__ unactivated_rotations;
        float scale_modifier;
        float** viewmatrix;
        float** projmatrix;
        glm::vec3* __restrict__ campos;
        float* tan_fovx;
        float* tan_fovy;
        float* cx;
        float* cy;
        float* focal_x;
        float* focal_y;
        bool have_n_contrib_vol_rend_prefix_sum;
        int** n_contrib_vol_rend_prefix_sum;
        bool have_residuals;
        float* residuals;
        bool have_weights;
        float* weights;
        bool have_residuals_ssim;
        float* residuals_ssim;
        bool have_weights_ssim;
        float* weights_ssim;
        int** map_visible_gaussians;
        int** map_cache_to_gaussians;
        int* n_visible_gaussians;
        bool debug;
        const int64_t offset_xyz, offset_scales, offset_rotations, offset_opacity, offset_features_dc, offset_features_rest;
        int* n_sparse_gaussians;

        // inputs created during construction
        int32_t max_n_sparse_gaussians;
        int32_t max_n_visible_gaussians;

        // pointers created for later usage
        bool pointers_changed = false;
        char** geomBuffer_ptrs;
        char** binningBuffer_ptrs;
        char** imageBuffer_ptrs;
        uint2** ranges_ptrs;
        int32_t** n_contrib_ptrs;
        float** accum_alpha_ptrs;
        uint32_t** point_list_ptrs;

        inline void allocate_pointer_memory();

        inline void free_pointer_memory() {
            cudaDeviceSynchronize();
            cudaFree(geomBuffer_ptrs);
            cudaFree(binningBuffer_ptrs);
            cudaFree(imageBuffer_ptrs);
            if(pointers_changed) {
                cudaFree(ranges_ptrs);
                cudaFree(n_contrib_ptrs);
                cudaFree(accum_alpha_ptrs);
                cudaFree(point_list_ptrs);
            }
        }
    };
}