#include <graphite/active.hpp>

namespace graphite {
size_t build_active_indices(const thrust::device_vector<uint8_t> &active,
                            thrust::device_vector<size_t> &active_indices,
                            const size_t count, const uint8_t level) {

  // Count active constraints
  const size_t active_count =
      thrust::count_if(thrust::device, active.begin(), active.end(),
                       [level] __device__(const uint8_t a) {
                         return is_factor_active(a, level);
                       });

  // Resize active indices to the number of active constraints
  active_indices.clear();
  active_indices.resize(active_count);

  // Fill active indices with the indices of active constraints
  thrust::copy_if(thrust::device, thrust::make_counting_iterator<size_t>(0),
                  thrust::make_counting_iterator<size_t>(count), active.begin(),
                  active_indices.begin(), [level] __device__(const uint8_t a) {
                    return is_factor_active(a, level);
                  });
  return active_count;
}
}