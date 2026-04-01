/// @file hessian.hpp
#pragma once
#include <graphite/block.hpp>
#include <graphite/common.hpp>
#include <graphite/csc_utils.hpp>
#include <graphite/stream.hpp>
#include <graphite/utils.hpp>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <unordered_map>
#include <utility>

namespace graphite {

template <typename T, typename I> class CSCMatrix {
public:
  thrust::device_vector<I> d_pointers;
  thrust::device_vector<I> d_indices;
  thrust::device_vector<T> d_values;
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

  void fill(const T &value) {
    thrust::fill(thrust::device, data.begin(), data.end(), value);
  }

  void zero() {
    thrust::fill(thrust::device, data.begin(), data.end(), static_cast<T>(0));
  }
};

template <typename T, typename S> class Hessian {
public:
  // Returns coordinates of upper triangular filled-in Hessian blocks
  std::vector<BlockCoordinates> get_block_coordinates(Graph<T, S> *graph) {
    // For a constraint to contribute to the Hessian,
    // the constraint must be active, and both variables must be active
    // (non-fixed), and the block must reside in the upper triangular part of
    // the Hessian (i.e., row index <= column index)
    thrust::device_vector<BlockCoordinates> block_coords;
    auto &f_desc = graph->get_factor_descriptors();
    for (auto &f : f_desc) {
      f->get_hessian_block_coordinates(block_coords);
    }

    thrust::sort(
        thrust::device, block_coords.begin(), block_coords.end(),
        [] __device__(const BlockCoordinates &a, const BlockCoordinates &b) {
          // sort so that we get blocks in column-major order (e.g consecutive
          // indices in same column)
          if (a.col == b.col) {
            return a.row < b.row;
          }
          return a.col < b.col;
        });

    // Remove duplicates
    auto end_it = thrust::unique(
        thrust::device, block_coords.begin(), block_coords.end(),
        [] __device__(const BlockCoordinates &a, const BlockCoordinates &b) {
          return (a.row == b.row) && (a.col == b.col);
        });

    // Resize the vector to remove duplicates
    block_coords.resize(end_it - block_coords.begin());

    // Copy to host and convert to std::vector
    std::vector<BlockCoordinates> host_block_coords(block_coords.size());
    thrust::copy(block_coords.begin(), block_coords.end(),
                 host_block_coords.begin());
    return host_block_coords;
  }

  // Block coords must be unique and in column-major order
  // Build block-csc style indices on the host that we can access on GPU for
  // quickly constructing a scalar CSC-style Hessian matrix (upper triangle
  // only)

  void build_indices(Graph<T, S> *graph,
                     const std::vector<BlockCoordinates> &block_coords) {
    const auto num_columns = graph->get_num_block_columns();
    csc::build_block_csc_indices(num_columns, block_indices, block_coords,
                                 d_col_pointers, d_row_indices, d_offsets);
    d_hessian_offsets = graph->get_offset_vector();
    csc::build_scalar_to_block_map(d_hessian_offsets, num_columns,
                                   scalar_to_block_map);
  }

  void backup_diagonal(Graph<T, S> *graph, StreamPool &streams) {
    d_prev_diagonal.resize(graph->get_hessian_dimension());

    auto diag = d_prev_diagonal.data().get();
    const auto h_offsets = d_hessian_offsets.data().get();
    const auto p_col = d_col_pointers.data().get();
    const auto r_idx = d_row_indices.data().get();
    const auto block_locations = d_offsets.data().get();
    const auto h = d_hessian.data().get();

    thrust::for_each(
        thrust::device, thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(graph->get_num_block_columns()),
        [=] __device__(size_t block_col) {
          const size_t hessian_col = h_offsets[block_col];
          const size_t dim = h_offsets[block_col + 1] - hessian_col;

          // find diagonal block in column where row == col
          // const auto start = p_col[block_col];
          const auto end = p_col[block_col + 1];

          const auto b = end - 1; // assume there is always a diagonal block and
                                  // it is always last

          if (r_idx[b] == block_col) {
            // found diagonal block, copy elements
            const auto block = h + block_locations[b];
            for (size_t i = 0; i < dim; i++) {
              diag[hessian_col + i] = block[i * dim + i];
            }
          }
        });
  }

  void apply_damping(Graph<T, S> *graph, T damping_factor,
                     const bool use_identity, StreamPool &streams) {
    // Assume diagonal was already backed up when recomputing Hessian values
    auto diag = d_prev_diagonal.data().get();
    const auto h_offsets = d_hessian_offsets.data().get();
    const auto p_col = d_col_pointers.data().get();
    const auto r_idx = d_row_indices.data().get();
    auto block_locations = d_offsets.data().get();
    auto h = d_hessian.data().get();

    thrust::for_each(
        thrust::device, thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(graph->get_num_block_columns()),
        [=] __device__(size_t block_col) {
          const size_t hessian_col = h_offsets[block_col];
          const size_t dim = h_offsets[block_col + 1] - hessian_col;

          // find diagonal block in column where row == col
          // const auto start = p_col[block_col];
          const auto end = p_col[block_col + 1];
          const auto b = end - 1; // assume there is always a diagonal block and
                                  // it is always last

          if (r_idx[b] == block_col) {
            // found diagonal block, backup then apply damping
            auto block = h + block_locations[b];
            for (size_t i = 0; i < dim; i++) {
              const auto h_diag = diag[hessian_col + i];
              if (use_identity) {
                block[i * dim + i] =
                    (S)((double)h_diag + (double)damping_factor);
              } else {
                block[i * dim + i] =
                    (S)((double)h_diag +
                        damping_factor *
                            std::clamp((double)h_diag, 1.0e-6, 1.0e32));
              }
            }
          }
        });
  }

  void setup_hessian_computation(Graph<T, S> *graph, StreamPool &streams) {
    // Build offset buffer for multiplying Jacobian blocks.
    // Each offset corresponds to the start index of an output Hessian block.\

    // Generate the offsets for each product, for each factor

    // First we need to compute the upper bound for the size of h_block_offsets
    const auto &factors = graph->get_factor_descriptors();
    size_t mul_count = 0;
    for (auto &f : factors) {
      const size_t num_vertices = f->get_num_descriptors();
      for (size_t i = 0; i < num_vertices; i++) {
        for (size_t j = i; j < num_vertices; j++) {
          mul_count += f->active_count();
        }
      }
    }

    h_block_offsets.resize(mul_count);

    // Now compute the offsets
    mul_count = 0;
    for (auto &f : factors) {
      mul_count += f->setup_hessian_computation(
          block_indices, d_hessian, h_block_offsets.data() + mul_count,
          streams);
    }

    // Final copy
    d_block_offsets = h_block_offsets;
  }

  void execute_hessian_computation(Graph<T, S> *graph, StreamPool &streams) {
    thrust::fill(thrust::device, d_hessian.begin(), d_hessian.end(),
                 static_cast<S>(0.0));
    size_t mul_count = 0;
    const auto &factors = graph->get_factor_descriptors();
    for (auto &f : factors) {
      mul_count += f->execute_hessian_computation(
          block_indices, d_hessian, d_block_offsets.data().get() + mul_count,
          streams);
    }
  }

  // Data
  std::unordered_map<BlockCoordinates, size_t> block_indices;
  thrust::device_vector<S> d_hessian;
  thrust::device_vector<size_t> d_col_pointers;
  thrust::device_vector<size_t> d_row_indices;
  thrust::device_vector<size_t> d_offsets;
  thrust::device_vector<S> d_prev_diagonal;
  thrust::device_vector<size_t> d_hessian_offsets;
  thrust::device_vector<size_t> scalar_to_block_map;

  // For calculating Hessian blocks
  thrust::host_vector<size_t> h_block_offsets;
  thrust::device_vector<size_t> d_block_offsets;

public:
  Hessian() = default;

  const thrust::device_vector<size_t> &get_block_col_pointers() const {
    return d_col_pointers;
  }

  const thrust::device_vector<size_t> &get_block_row_indices() const {
    return d_row_indices;
  }

  const thrust::device_vector<size_t> &get_block_value_offsets() const {
    return d_offsets;
  }

  const thrust::device_vector<S> &get_values() const { return d_hessian; }

  S *get_values_ptr() { return d_hessian.data().get(); }

  const S *get_values_ptr() const { return d_hessian.data().get(); }

  void build_structure(Graph<T, S> *graph, StreamPool &streams) {
    // Implementation for building the Hessian matrix

    // Assume we don't have a GPU hash map.
    // First we need to count how many Hessian blocks we have
    // We'll create a set of Hessian block coordinates by iterating over
    // descriptors Ignore blocks not in the upper triangular part
    const auto block_coords = get_block_coordinates(graph);

    // Then we need to allocate memory for each block
    // We can iterate the set and figure out the total memory,
    // then allocate a big chunk and assign pointers accordingly
    // TODO: Maybe we can use an GPU exclusive scan instead?
    size_t num_values = 0;
    block_indices.clear();
    block_indices.reserve(block_coords.size());
    for (const auto &coord : block_coords) {
      block_indices[coord] = num_values;
      num_values += graph->get_variable_dimension(coord.row) *
                    graph->get_variable_dimension(coord.col);
    }
    d_hessian.resize(num_values);

    // We need to end up with a block CSC-style representation
    // where we can iterate down the blocks in each block columnn
    // and retrieve the data pointer for each block for the purpose of
    // constructing a scalar CSC-style representation.
    build_indices(graph, block_coords);

    // Setup data for computing Hessian as product of Jacobians
    setup_hessian_computation(graph, streams);
  }

  void update_values(Graph<T, S> *graph, StreamPool &streams) {
    // thrust::fill(thrust::device, d_hessian.begin(), d_hessian.end(),
    //              static_cast<S>(0.0));

    // setup_hessian_computation(graph, streams);
    execute_hessian_computation(graph, streams);

    // auto &f_desc = graph->get_factor_descriptors();
    // for (auto &f : f_desc) {
    //   h_block_offsets.clear();
    //   d_block_offsets.clear();
    //   f->compute_hessian_blocks(block_indices, d_hessian, h_block_offsets,
    //                             d_block_offsets, streams);
    // }

    // Back up diagonal for damping purposes
    backup_diagonal(graph, streams);
  }

  template <typename I>
  void build_csc_structure(Graph<T, S> *graph, CSCMatrix<S, I> &matrix) {
    const auto hessian_dim = graph->get_hessian_dimension();
    csc::build_scalar_csc_structure<S, I>(hessian_dim, d_col_pointers,
                                          d_row_indices, d_hessian_offsets,
                                          scalar_to_block_map, matrix);
  }

  template <typename I>
  void update_csc_values(Graph<T, S> *graph, CSCMatrix<S, I> &matrix) {
    const auto hessian_dim = graph->get_hessian_dimension();
    csc::update_scalar_csc_values<S, I>(
        hessian_dim, d_hessian, d_col_pointers, d_row_indices, d_offsets,
        d_hessian_offsets, scalar_to_block_map, matrix);
  }
};

} // namespace graphite