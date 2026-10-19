// Nonnegative SIRT for the dose operator B=A^T. Both products use float32 CSR.
// Python owns all matrices and workspaces; no allocation scales with volume Z.
#include <algorithm>
#include <cstddef>
#include <vector>

#ifdef _WIN32
#define SPIN_SLICER_EXPORT __declspec(dllexport)
#else
#define SPIN_SLICER_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

SPIN_SLICER_EXPORT void forward_project(
    int num_rows, int num_columns, int nz, const float* values,
    const int* indptr, const int* indices, const float* data, float* result
) {
    (void)num_columns;
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < num_rows; ++row) {
        float* destination = result + static_cast<std::size_t>(row) * nz;
        std::fill(destination, destination + nz, 0.0f);
        for (int index = indptr[row]; index < indptr[row + 1]; ++index) {
            const float* source = values + static_cast<std::size_t>(indices[index]) * nz;
            const float weight = data[index];
            #pragma omp simd
            for (int z = 0; z < nz; ++z) {
                destination[z] += weight * source[z];
            }
        }
    }
}

SPIN_SLICER_EXPORT void sirt_linear_step(
    int num_rays, int num_pixels, int nz, float* projections, const float* target,
    const int* a_indptr, const int* a_indices, const float* a_data,
    const int* at_indptr, const int* at_indices, const float* at_data,
    const float* ray_norm, const float* voxel_norm,
    const bool* hits, float* workspace
) {
    forward_project(num_pixels, num_rays, nz, projections, at_indptr, at_indices, at_data, workspace);
    #pragma omp parallel for schedule(static)
    for (int pixel = 0; pixel < num_pixels; ++pixel) {
        const std::size_t offset = static_cast<std::size_t>(pixel) * nz;
        #pragma omp simd
        for (int z = 0; z < nz; ++z) {
            workspace[offset + z] = (target[offset + z] - workspace[offset + z]) * voxel_norm[pixel];
        }
    }

    #pragma omp parallel
    {
        std::vector<float> update(nz);
        #pragma omp for schedule(static)
        for (int ray = 0; ray < num_rays; ++ray) {
            std::fill(update.begin(), update.end(), 0.0f);
            for (int index = a_indptr[ray]; index < a_indptr[ray + 1]; ++index) {
                const float* error = workspace + static_cast<std::size_t>(a_indices[index]) * nz;
                const float weight = a_data[index];
                #pragma omp simd
                for (int z = 0; z < nz; ++z) {
                    update[z] += weight * error[z];
                }
            }
            const std::size_t offset = static_cast<std::size_t>(ray) * nz;
            #pragma omp simd
            for (int z = 0; z < nz; ++z) {
                projections[offset + z] = hits[offset + z]
                    ? std::max(0.0f, projections[offset + z] + ray_norm[ray] * update[z])
                    : 0.0f;
            }
        }
    }
}

} // extern "C"
