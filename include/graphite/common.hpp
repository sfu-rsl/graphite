#pragma once
#include <array>
#include <boost/functional/hash.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include <memory>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>
// #include <tuple>
#include <cuda/std/tuple>

#include <glso/dual.hpp>

// injection for hashing std::pair<size_t, size_t>
namespace std {
template <> struct hash<std::pair<size_t, size_t>> {
  size_t operator()(const std::pair<size_t, size_t> &p) const {
    size_t seed = 0;
    boost::hash_combine(seed, p.first);
    boost::hash_combine(seed, p.second);
    return seed;
    // return std::hash<size_t>()(p.first) ^ std::hash<size_t>()(p.second);
  }
};
} // namespace std

namespace glso {
#define hd_fn __host__ __device__
} // namespace glso