#pragma once
#include <array>
#include <vector>
#include <utility>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <memory>
// #include <tuple>
#include <cuda/std/tuple>

#include <glso/dual.hpp>

// injection for hashing std::pair<size_t, size_t>
namespace std {
    template <>
    struct hash<std::pair<size_t, size_t>> {
        size_t operator()(const std::pair<size_t, size_t>& p) const {
            size_t seed = 0;
            boost::hash_combine(seed, p.first);
            boost::hash_combine(seed, p.second);
            return seed;
            // return std::hash<size_t>()(p.first) ^ std::hash<size_t>()(p.second);
        }
    };
}