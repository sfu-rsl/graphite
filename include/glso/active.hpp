#pragma once
#include <thrust/count.h>
#include <thrust/copy.h>

namespace glso {

    __host__ __device__ bool is_active(const uint8_t active_val, const uint8_t level) {
        constexpr uint8_t NOT_MSB = 0x7F; // 01111111
        return (active_val & NOT_MSB) > level;
    }

    // __global__ void initialize_active_indices(
    //     const uint8_t *active,
    //     size_t *active_indices,
    //     const size_t count,
    //     const uint8_t level
    // ) {
    //     const size_t local_id = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
    //      static_cast<size_t>(threadIdx.x);

    //      if (local_id >= count) {
    //         return; // Out of bounds
    //      }

        

    // }

    // Returns the number of active constraints and fills the active_indices vector
    size_t build_active_indices(
        const thrust::device_vector<uint8_t> &active,
        thrust::device_vector<size_t> &active_indices,
        const size_t count,
        const uint8_t level
    ) {

        constexpr uint8_t NOT_MSB = 0x7F; // 01111111

        // Count active constraints
        const size_t active_count = thrust::count_if(
            thrust::device,
            active.begin(),
            active.end(),
            [level] __device__(const uint8_t a) { 
                return is_active(a, level);
            }
        );

        // Resize active indices to the number of active constraints
        active_indices.clear();
        active_indices.reserve(active_count);

        // Fill active indices with the indices of active constraints
        thrust::copy_if(
            thrust::device,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(count),
            active_indices.begin(),
            [level] __device__(const uint8_t a) { 
                return is_active(a, level);
            }
        );
        return active_count;
    }

}