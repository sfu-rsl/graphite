/// @file csc_utils.hpp
#pragma once

#include <graphite/block.hpp>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <unordered_map>
#include <vector>

namespace graphite {
namespace csc {

inline void build_block_csc_indices(
    size_t num_block_cols,
    const std::unordered_map<BlockCoordinates, size_t> &block_indices,
    const std::vector<BlockCoordinates> &block_coords,
    thrust::device_vector<size_t> &d_col_pointers,
    thrust::device_vector<size_t> &d_row_indices,
    thrust::device_vector<size_t> &d_offsets) {
  std::vector<BlockCoordinates> sorted = block_coords;
  std::sort(sorted.begin(), sorted.end(),
            [](const BlockCoordinates &a, const BlockCoordinates &b) {
              if (a.col == b.col) {
                return a.row < b.row;
              }
              return a.col < b.col;
            });

  thrust::host_vector<size_t> h_col_pointers(num_block_cols + 1);
  thrust::host_vector<size_t> h_row_indices(sorted.size());
  thrust::host_vector<size_t> h_offsets(sorted.size());
  thrust::fill(thrust::host, h_col_pointers.begin(), h_col_pointers.end(), 0);

  for (size_t i = 0; i < sorted.size(); i++) {
    const auto &coord = sorted[i];
    h_col_pointers[coord.col]++;
    h_row_indices[i] = coord.row;
    h_offsets[i] = block_indices.at(coord);
  }

  d_col_pointers = h_col_pointers;
  d_row_indices = h_row_indices;
  d_offsets = h_offsets;

  thrust::exclusive_scan(thrust::device, d_col_pointers.begin(),
                         d_col_pointers.end(), d_col_pointers.begin());
}

inline void
build_scalar_to_block_map(const thrust::device_vector<size_t> &d_scalar_offsets,
                          size_t num_block_cols,
                          thrust::device_vector<size_t> &scalar_to_block_map) {
  const size_t scalar_dim = d_scalar_offsets[num_block_cols];
  scalar_to_block_map.resize(scalar_dim);

  auto s2b_map = scalar_to_block_map.data().get();
  const auto offsets = d_scalar_offsets.data().get();

  thrust::for_each(thrust::device, thrust::make_counting_iterator<size_t>(0),
                   thrust::make_counting_iterator<size_t>(num_block_cols),
                   [=] __device__(size_t block_col) {
                     const size_t scalar_col = offsets[block_col];
                     const size_t dim = offsets[block_col + 1] - scalar_col;
                     for (size_t i = 0; i < dim; i++) {
                       s2b_map[scalar_col + i] = block_col;
                     }
                   });
}

template <typename S, typename I, typename Matrix>
void build_scalar_csc_structure(
    size_t scalar_dim, const thrust::device_vector<size_t> &d_col_pointers,
    const thrust::device_vector<size_t> &d_row_indices,
    const thrust::device_vector<size_t> &d_scalar_offsets,
    const thrust::device_vector<size_t> &scalar_to_block_map, Matrix &matrix) {
  matrix.d_pointers.resize(scalar_dim + 1);
  thrust::fill(thrust::device, matrix.d_pointers.begin(),
               matrix.d_pointers.end(), 0);

  const auto p_col = d_col_pointers.data().get();
  const auto r_idx = d_row_indices.data().get();
  const auto scalar_offsets = d_scalar_offsets.data().get();
  const auto s2b_map = scalar_to_block_map.data().get();
  auto scalar_ptrs = matrix.d_pointers.data().get();

  thrust::for_each(thrust::device, thrust::make_counting_iterator<size_t>(0),
                   thrust::make_counting_iterator<size_t>(scalar_dim),
                   [=] __device__(const size_t scalar_col) {
                     const size_t block_col = s2b_map[scalar_col];
                     const size_t start = p_col[block_col];
                     const size_t end = p_col[block_col + 1];
                     size_t num_values = 0;

                     for (size_t b = start; b < end; b++) {
                       const size_t block_row = r_idx[b];
                       const size_t nrows = scalar_offsets[block_row + 1] -
                                            scalar_offsets[block_row];
                       const size_t scalar_row = scalar_offsets[block_row];

                       for (size_t r = 0; r < nrows; r++) {
                         if (scalar_row + r <= scalar_col) {
                           num_values++;
                         } else {
                           break;
                         }
                       }
                     }
                     scalar_ptrs[scalar_col] = num_values;
                   });

  thrust::exclusive_scan(thrust::device, matrix.d_pointers.begin(),
                         matrix.d_pointers.end(), matrix.d_pointers.begin());

  const size_t nnz = matrix.d_pointers[scalar_dim];
  matrix.d_indices.resize(nnz);
  matrix.d_values.resize(nnz);

  auto row_indices = matrix.d_indices.data().get();

  thrust::for_each(thrust::device, thrust::make_counting_iterator<size_t>(0),
                   thrust::make_counting_iterator<size_t>(scalar_dim),
                   [=] __device__(const size_t scalar_col) {
                     const size_t block_col = s2b_map[scalar_col];
                     const size_t start = p_col[block_col];
                     const size_t end = p_col[block_col + 1];
                     size_t write_idx = scalar_ptrs[scalar_col];

                     for (size_t b = start; b < end; b++) {
                       const size_t block_row = r_idx[b];
                       const size_t nrows = scalar_offsets[block_row + 1] -
                                            scalar_offsets[block_row];
                       const size_t scalar_row = scalar_offsets[block_row];

                       for (size_t r = 0; r < nrows; r++) {
                         if (scalar_row + r <= scalar_col) {
                           row_indices[write_idx++] =
                               static_cast<I>(scalar_row + r);
                         } else {
                           break;
                         }
                       }
                     }
                   });
}

template <typename S, typename I, typename Matrix>
void update_scalar_csc_values(
    size_t scalar_dim, const thrust::device_vector<S> &block_values,
    const thrust::device_vector<size_t> &d_col_pointers,
    const thrust::device_vector<size_t> &d_row_indices,
    const thrust::device_vector<size_t> &d_offsets,
    const thrust::device_vector<size_t> &d_scalar_offsets,
    const thrust::device_vector<size_t> &scalar_to_block_map, Matrix &matrix) {
  const auto values = block_values.data().get();
  const auto p_col = d_col_pointers.data().get();
  const auto r_idx = d_row_indices.data().get();
  const auto block_offsets = d_offsets.data().get();
  const auto scalar_offsets = d_scalar_offsets.data().get();
  const auto s2b_map = scalar_to_block_map.data().get();

  const auto scalar_ptrs = matrix.d_pointers.data().get();
  auto out_values = matrix.d_values.data().get();

  thrust::for_each(
      thrust::device, thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(scalar_dim),
      [=] __device__(const size_t scalar_col) {
        const size_t block_col = s2b_map[scalar_col];
        const size_t start = p_col[block_col];
        const size_t end = p_col[block_col + 1];
        size_t write_idx = scalar_ptrs[scalar_col];

        for (size_t b = start; b < end; b++) {
          const size_t block_row = r_idx[b];
          const size_t nrows =
              scalar_offsets[block_row + 1] - scalar_offsets[block_row];
          const size_t scalar_row = scalar_offsets[block_row];
          const size_t col_in_block = scalar_col - scalar_offsets[block_col];

          const auto block = values + block_offsets[b] + col_in_block * nrows;
          for (size_t r = 0; r < nrows; r++) {
            if (scalar_row + r <= scalar_col) {
              out_values[write_idx++] = block[r];
            } else {
              break;
            }
          }
        }
      });
}

} // namespace csc
} // namespace graphite
