#include <vector>
#include <cmath>
#include <algorithm>

#ifdef _WIN32
#define SPIN_SLICER_EXPORT __declspec(dllexport)
#else
#define SPIN_SLICER_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

SPIN_SLICER_EXPORT void forward_project(
    int num_rays,
    int num_pixels,
    int nz,
    const float* V_target,         // [num_pixels, nz]
    const int* A_indptr,           // length num_rays + 1
    const int* A_indices,          // length nnz
    const float* A_data,           // length nnz
    float* P0                      // [num_rays, nz]
) {
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < num_rays; ++r) {
        int start = A_indptr[r];
        int end = A_indptr[r+1];
        
        // Initialize P0[r, :] to 0
        for (int z = 0; z < nz; ++z) {
            P0[r * nz + z] = 0.0f;
        }
        
        for (int idx = start; idx < end; ++idx) {
            int p = A_indices[idx];
            float val = A_data[idx];
            const float* v_ptr = &V_target[p * nz];
            float* p_ptr = &P0[r * nz];
            
            #pragma omp simd
            for (int z = 0; z < nz; ++z) {
                p_ptr[z] += val * v_ptr[z];
            }
        }
    }
}

SPIN_SLICER_EXPORT void sirt_loop(
    int num_rays,
    int num_pixels,
    int nz,
    int iterations,
    float d_h,
    const float* b_flat,           // [num_rays, nz]
    const float* V_target,         // [num_pixels, nz]
    const int* A_indptr,           // length num_rays + 1
    const int* A_indices,          // length nnz
    const float* A_data,           // length nnz
    const int* AT_indptr,          // length num_pixels + 1
    const int* AT_indices,         // length nnz
    const float* AT_data,          // length nnz
    const float* R_norm,           // length num_rays
    const float* C_norm,           // length num_pixels
    const bool* ray_hits_any,      // length num_rays
    float* b_opt                   // [num_rays, nz]
) {
    #pragma omp parallel for
    for (long long i = 0; i < (long long)num_rays * nz; ++i) {
        b_opt[i] = b_flat[i];
    }

    std::vector<float> dose(num_pixels * nz, 0.0f);
    std::vector<float> error(num_pixels * nz, 0.0f);

    for (int it = 0; it < iterations; ++it) {
        // 1. Reconstruct delivered dose in resin vat: dose = AT.dot(b_opt)
        #pragma omp parallel for schedule(dynamic)
        for (int p = 0; p < num_pixels; ++p) {
            int start = AT_indptr[p];
            int end = AT_indptr[p+1];
            
            for (int z = 0; z < nz; ++z) {
                dose[p * nz + z] = 0.0f;
            }
            
            for (int idx = start; idx < end; ++idx) {
                int r = AT_indices[idx];
                float val = AT_data[idx];
                const float* b_ptr = &b_opt[r * nz];
                float* d_ptr = &dose[p * nz];
                
                #pragma omp simd
                for (int z = 0; z < nz; ++z) {
                    d_ptr[z] += val * b_ptr[z];
                }
            }
        }

        // 2. Normalize dose globally
        float dose_max = 0.0f;
        #pragma omp parallel for reduction(max:dose_max)
        for (long long i = 0; i < (long long)num_pixels * nz; ++i) {
            if (dose[i] > dose_max) {
                dose_max = dose[i];
            }
        }
        if (dose_max <= 0.0f) {
            dose_max = 1.0f;
        }

        float inv_dose_max = 1.0f / dose_max;

        // 3. Sigmoid threshold response & error computation
        #pragma omp parallel for simd
        for (long long i = 0; i < (long long)num_pixels * nz; ++i) {
            float dose_norm = dose[i] * inv_dose_max;
            float dose_arg = -15.0f * (dose_norm - d_h);
            if (dose_arg < -80.0f) dose_arg = -80.0f;
            if (dose_arg > 80.0f) dose_arg = 80.0f;
            float dose_thresh = 1.0f / (1.0f + std::exp(dose_arg));
            float sigmoid_deriv = 15.0f * dose_thresh * (1.0f - dose_thresh);
            // Add a small epsilon to avoid zero gradients on saturated regions
            if (sigmoid_deriv < 0.05f) sigmoid_deriv = 0.05f;
            error[i] = (V_target[i] - dose_thresh) * sigmoid_deriv;
        }

        // 4. Update rule: delta_b = A.dot(C_norm[:, None] * error)
        #pragma omp parallel
        {
            std::vector<float> delta(nz, 0.0f);

            #pragma omp for schedule(dynamic)
            for (int r = 0; r < num_rays; ++r) {
                if (!ray_hits_any[r]) {
                    for (int z = 0; z < nz; ++z) {
                        b_opt[r * nz + z] = 0.0f;
                    }
                    continue;
                }

                int start = A_indptr[r];
                int end = A_indptr[r+1];
                float r_norm_val = R_norm[r];

                std::fill(delta.begin(), delta.end(), 0.0f);

                for (int idx = start; idx < end; ++idx) {
                    int p = A_indices[idx];
                    float val = A_data[idx];
                    float c_norm_val = C_norm[p];
                    float combined_val = val * c_norm_val;
                    const float* err_ptr = &error[p * nz];
                    
                    #pragma omp simd
                    for (int z = 0; z < nz; ++z) {
                        delta[z] += combined_val * err_ptr[z];
                    }
                }
                
                float* b_ptr = &b_opt[r * nz];
                #pragma omp simd
                for (int z = 0; z < nz; ++z) {
                    float new_val = b_ptr[z] + r_norm_val * delta[z];
                    if (new_val < 0.0f) new_val = 0.0f;
                    b_ptr[z] = new_val;
                }
            }
        }
    }
}

} // extern "C"
