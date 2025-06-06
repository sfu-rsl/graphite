#pragma once
#include <glso/factor.hpp>
#include <glso/vertex.hpp>
#include <glso/visitor.hpp>
#include <limits>
#include <thrust/execution_policy.h>

namespace glso {

// template<typename T> class Solver;

class BlockCoordinates {
public:
  size_t row;
  size_t col;
};

template <typename T> class HessianBlocks {
public:
  std::pair<size_t, size_t> dimensions;
  size_t num_blocks;
  thrust::device_vector<T> data;
  thrust::device_vector<BlockCoordinates> block_coordinates;

  void resize(size_t num_blocks, size_t rows, size_t cols) {
    dimensions = {rows, cols};
    this->num_blocks = num_blocks;
    data.resize(rows * cols * num_blocks);
    block_coordinates.resize(num_blocks);
  }
};

template <typename T, typename S> class Graph {

private:
  GraphVisitor<T, S> visitor;
  std::vector<BaseVertexDescriptor<T, S> *> vertex_descriptors;
  std::vector<BaseFactorDescriptor<T, S> *> factor_descriptors;
  thrust::device_vector<T> b;
  thrust::device_vector<T> jacobian_scales;
  size_t hessian_column;

public:
  Graph() {}

  size_t get_hessian_dimension() { return hessian_column; }

  thrust::device_vector<T> &get_b() { return b; }

  std::vector<BaseVertexDescriptor<T, S> *> &get_vertex_descriptors() {
    return vertex_descriptors;
  }

  std::vector<BaseFactorDescriptor<T, S> *> &get_factor_descriptors() {
    return factor_descriptors;
  }

  void add_vertex_descriptor(BaseVertexDescriptor<T, S> *descriptor) {
    vertex_descriptors.push_back(descriptor);
  }

  template <typename F> void add_factor_descriptor(F *factor) {
    factor_descriptors.push_back(factor);
  }

  bool initialize_optimization() {

    // For each vertex descriptor, take global to local id mapping and transform
    // it into a Hessian column to local id mapping.

    std::vector<std::pair<size_t, std::pair<size_t, size_t>>>
        global_to_local_combined;

    for (size_t i = 0; i < vertex_descriptors.size(); ++i) {
      const auto &map = vertex_descriptors[i]->get_global_map();
      for (const auto &entry : map) {
        global_to_local_combined.push_back({entry.first, {i, entry.second}});
      }
    }

    // Sort the combined list by global ID
    std::sort(global_to_local_combined.begin(), global_to_local_combined.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    // Assign Hessian columns to local indices
    hessian_column = 0;
    for (const auto &entry : global_to_local_combined) {
      if (!vertex_descriptors[entry.second.first]->is_fixed(entry.first)) {
        vertex_descriptors[entry.second.first]->set_hessian_column(
            entry.first, hessian_column);
        hessian_column += vertex_descriptors[entry.second.first]->dimension();
      }
    }

    // Copy vertex values to device
    for (auto &desc : vertex_descriptors) {
      desc->to_device();
    }

    // Copy factors to device
    for (auto &desc : factor_descriptors) {
      desc->to_device();
    }

    // Initialize Jacobian storage
    for (auto &f : factor_descriptors) {
      f->initialize_jacobian_storage();
    }

    return true;
  }

  bool build_structure() {
    // Allocate storage for solver vectors
    const auto size_x = get_hessian_dimension();
    b.resize(size_x);
    jacobian_scales.resize(size_x);

    return true;
  }

  void compute_error() {
    for (auto &factor : factor_descriptors) {
      factor->visit_error(visitor); // TODO: Make non-autodiff version
    }
    cudaDeviceSynchronize();
  }

  T chi2() {
    T chi2 = static_cast<T>(0);
    for (auto &factor : factor_descriptors) {
      chi2 += factor->chi2(visitor);
    }
    return chi2;
  }

  void linearize() {

    for (auto &factor : factor_descriptors) {
      // compute error
      if (factor->use_autodiff()) {
        factor->visit_error_autodiff(visitor);
      } else {
        factor->visit_error(visitor);
        // manually compute Jacobians
        throw std::runtime_error("Manual Jacobian computation not implemented");
      }
    }

    cudaDeviceSynchronize();

    // Compute chi2
    chi2();

    // Compute Jacobian scale
    constexpr bool scale_jacobians = true;
    if (scale_jacobians) {
      thrust::fill(jacobian_scales.begin(), jacobian_scales.end(), 0);
      for (auto &factor : factor_descriptors) {
        factor->visit_scalar_diagonal(visitor, jacobian_scales.data().get());
      }
      cudaDeviceSynchronize();

      thrust::transform(
          thrust::device, jacobian_scales.begin(), jacobian_scales.end(),
          jacobian_scales.begin(), [] __device__(T value) {
            // return 1.0 / (1.0 + sqrt(value));
            constexpr double num = 1.0;
            const double denom = std::numeric_limits<double>::epsilon() +
                                 sqrt(static_cast<double>(value));
            // const double denom = 1.0
            //   + sqrt(static_cast<double>(value));
            // return 1.0 / (std::numeric_limits<T>::epsilon() + sqrt(value));
            return static_cast<T>(num / denom);
          });
    } else {
      thrust::fill(jacobian_scales.begin(), jacobian_scales.end(), 1.0);
    }

    // Scale Jacobians
    for (auto &factor : factor_descriptors) {
      factor->scale_jacobians(visitor, jacobian_scales.data().get());
    }
    cudaDeviceSynchronize();

    // Calculate b=J^T * r
    thrust::fill(b.begin(), b.end(), 0);
    for (auto &fd : factor_descriptors) {
      fd->visit_b(visitor, b.data().get());
    }

    cudaDeviceSynchronize();
  }

  void apply_step(const T *delta_x) {
    for (auto &desc : vertex_descriptors) {
      desc->visit_update(visitor, delta_x, jacobian_scales.data().get());
    }
    cudaDeviceSynchronize();
  }

  void backup_parameters() {

    for (const auto &desc : vertex_descriptors) {
      desc->backup_parameters();
    }

    cudaDeviceSynchronize();
  }

  void revert_parameters() {

    for (auto &desc : vertex_descriptors) {
      desc->restore_parameters();
    }

    cudaDeviceSynchronize();
  }

  void to_host() {
    for (auto &desc : vertex_descriptors) {
      desc->to_host();
    }
  }
};

} // namespace glso