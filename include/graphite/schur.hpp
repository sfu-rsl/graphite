/// @file schur.hpp
#pragma once

#include "block.hpp"
#include "graphite/block.hpp"
#include <algorithm>
#include <boost/container_hash/hash.hpp>
#include <chrono>
#include <cstdint>
#include <cublas_v2.h>
#include <graphite/csc_utils.hpp>
#include <graphite/graph.hpp>
#include <graphite/hessian.hpp>
#include <graphite/ops/schur.hpp>
#include <graphite/ops/vector.hpp>
#include <graphite/stream.hpp>
#include <iostream>
#include <limits>
#include <ratio>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>
#include <unordered_map>
#include <vector>

namespace graphite {

// Triplet dimensions for operation D += A*B*C, where B is always a square
// block.
struct ProductDim {
  size_t dim_a;
  size_t dim_b;
  size_t dim_c;

  bool operator==(const ProductDim &other) const {
    return dim_a == other.dim_a && dim_b == other.dim_b && dim_c == other.dim_c;
  }
};

using MatVecDim = BlockDimension;

struct PairCountFromPoseCount {
  __host__ __device__ size_t operator()(size_t count) const {
    return count * (count + 1) / 2;
  }
};

struct BlockCoordColMajorLess {
  __host__ __device__ bool operator()(const BlockCoordinates &a,
                                      const BlockCoordinates &b) const {
    if (a.col == b.col) {
      return a.row < b.row;
    }
    return a.col < b.col;
  }
};

} // namespace graphite

namespace std {
template <> struct hash<graphite::ProductDim> {
  size_t operator()(const graphite::ProductDim &pd) const {
    size_t seed = 0;
    boost::hash_combine(seed, pd.dim_a);
    boost::hash_combine(seed, pd.dim_b);
    boost::hash_combine(seed, pd.dim_c);
    return seed;
  }
};

} // namespace std

namespace graphite {

/**
 * @brief Class for computing the explicit Schur complement.
 */
template <typename T, typename S> class SchurComplement {
public:
  SchurComplement(Hessian<T, S> &H) : H(H) {
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    lowest_eliminated_block_col = std::numeric_limits<size_t>::max();
  }
  ~SchurComplement() { cublasDestroy(handle); }

  // Important dimensions and offsets
  size_t lowest_eliminated_block_col;
  size_t pose_col_start;
  size_t landmark_col_start;
  size_t num_block_columns;
  size_t pose_dim;
  size_t landmark_dim;
  thrust::device_vector<S> values;
  using P = std::conditional_t<is_low_precision<S>::value, T,
                               S>; // from block Jacobi preconditioner: don't
                                   // want to invert at low precision
  // let's actually disable the low precision path for now
  static_assert(
      !is_low_precision<S>::value,
      "SchurComplement does not currently support low precision types.");
  static_assert(
      std::is_same<T, S>::value,
      "SchurComplement currently requires T and S to be the same type.");
  thrust::device_vector<P>
      diagonal_values; // stores Hll block diagonals, then overwritten in-place
                       // with Hll^(-1).
  Hessian<T, S> &H;
  std::unordered_map<BlockCoordinates, size_t> block_indices;
  std::unordered_map<BlockCoordinates, size_t>
      diagonal_indices; // block diagonal indices for Hll^(-1)
  std::unordered_map<size_t, std::vector<BlockCoordinates>>
      diagonal_coords_by_dim;

  // Buffers for multiplying the three matrices
  std::unordered_map<ProductDim, thrust::device_vector<ops::MulOp<S>>> mul_ops;
  std::unordered_map<ProductDim, std::vector<ops::MulOp<S>>> h_mul_ops;
  std::unordered_map<MatVecDim, thrust::device_vector<ops::HplMatVecOp>>
      hpl_vec_ops;
  std::unordered_map<MatVecDim, std::vector<ops::HplMatVecOp>> h_hpl_vec_ops;
  std::unordered_map<MatVecDim, thrust::device_vector<ops::HplMatVecOp>>
      hplt_vec_ops;
  std::unordered_map<MatVecDim, std::vector<ops::HplMatVecOp>> h_hplt_vec_ops;
  std::unordered_map<MatVecDim, thrust::device_vector<ops::BlockCopyOp>>
      hpp_copy_ops;
  std::unordered_map<MatVecDim, std::vector<ops::BlockCopyOp>> h_hpp_copy_ops;

  // Per-block-size buffers for cublas*matinvBatched.
  thrust::host_vector<P *> A_ptrs;
  thrust::host_vector<P *> Ainv_ptrs;
  std::unordered_map<size_t, thrust::device_vector<P *>> A_ptrs_device_by_dim;
  std::unordered_map<size_t, thrust::device_vector<P *>>
      Ainv_ptrs_device_by_dim;
  std::unordered_map<size_t, thrust::device_vector<int>> info_by_dim;
  std::unordered_map<size_t, thrust::device_vector<size_t>>
      diagonal_value_offsets_by_dim;
  std::unordered_map<size_t, thrust::device_vector<size_t>>
      landmark_local_offsets_by_dim;
  thrust::host_vector<size_t> h_block_col_pointers;
  thrust::host_vector<size_t> h_block_row_indices;
  thrust::host_vector<size_t> h_block_offsets;

  // Block-CSC style metadata for Schur matrix blocks.
  thrust::device_vector<size_t> d_col_pointers;
  thrust::device_vector<size_t> d_row_indices;
  thrust::device_vector<size_t> d_offsets;
  thrust::device_vector<size_t> d_schur_offsets;
  thrust::device_vector<size_t> scalar_to_block_map;

  // Schur and temporary device vectors
  thrust::device_vector<T> b_Schur;
  thrust::device_vector<T> l_workspace;
  thrust::device_vector<T> p_workspace;
  thrust::device_vector<T> diagonal_workspace;

  // For Schur structure
  thrust::device_vector<size_t> d_pose_counts;
  thrust::device_vector<size_t> d_pair_counts;
  thrust::device_vector<size_t> d_pair_offsets;
  thrust::device_vector<BlockCoordinates> d_pairs;
  thrust::host_vector<BlockCoordinates> h_pairs;
  thrust::device_vector<ops::SchurMulTuple> d_mul_tuples;
  thrust::host_vector<ops::SchurMulTuple> h_mul_tuples;

  // For Schur multiplication
  std::vector<size_t> block_dims;

  // For diagonal inverse multiplication
  thrust::host_vector<size_t> h_diagonal_offsets;
  thrust::host_vector<size_t> h_local_offsets;

  // For building indices
  thrust::host_vector<size_t> h_schur_offsets;
  std::vector<BlockCoordinates> block_coords;

  cublasHandle_t handle;

  void build_structure(Graph<T, S> *graph, StreamPool &streams) {
    h_block_col_pointers = H.get_block_col_pointers();
    h_block_row_indices = H.get_block_row_indices();
    h_block_offsets = H.get_block_value_offsets();
    const auto &vertex_descriptors = graph->get_vertex_descriptors();

    lowest_eliminated_block_col = graph->get_elimination_block_column();
    pose_col_start = 0;
    landmark_col_start = lowest_eliminated_block_col;
    num_block_columns = graph->get_num_block_columns();
    pose_dim = graph->get_offset_vector()[landmark_col_start];
    landmark_dim = graph->get_hessian_dimension() - pose_dim;
    b_Schur.resize(pose_dim);

    build_schur_structure(graph, streams);

    setup_Hpp_copy(graph, streams);

    build_indices(graph, streams);

    build_diagonal_structure(graph, streams);

    setup_schur_multiplication(graph, streams);

    setup_diagonal_inverse_multiply(graph, streams);

    setup_Hpl_vector_multiply(graph, streams);

    setup_HplT_vector_multiply(graph, streams);

    setup_b_Schur_computation(graph, streams);
  }

  void update_values(Graph<T, S> *graph, StreamPool &streams) {
    execute_Hpp_copy(graph, streams);

    execute_block_diagonal_inversion(graph, streams);

    execute_schur_multiplication(graph, streams);

    execute_b_Schur_computation(graph, streams, b_Schur.data().get());
  }

  thrust::device_vector<T> &get_b_Schur() { return b_Schur; }

  void build_indices(Graph<T, S> *graph, StreamPool &streams) {
    (void)streams;
    block_coords.clear();
    block_coords.reserve(block_indices.size());
    for (const auto &entry : block_indices) {
      block_coords.push_back(entry.first);
    }

    csc::build_block_csc_indices(landmark_col_start, block_indices,
                                 block_coords, d_col_pointers, d_row_indices,
                                 d_offsets);

    // Build scalar offsets for the Schur block columns (p-part only).
    h_schur_offsets.resize(landmark_col_start + 1);
    h_schur_offsets[0] = 0;
    for (size_t bc = 0; bc < landmark_col_start; bc++) {
      h_schur_offsets[bc + 1] =
          h_schur_offsets[bc] + graph->get_variable_dimension(bc);
    }
    d_schur_offsets = h_schur_offsets;
    csc::build_scalar_to_block_map(d_schur_offsets, landmark_col_start,
                                   scalar_to_block_map);
  }

  template <typename I>
  void build_csc_structure(Graph<T, S> *graph, CSCMatrix<S, I> &matrix) {
    (void)graph;
    csc::build_scalar_csc_structure<S, I>(pose_dim, d_col_pointers,
                                          d_row_indices, d_schur_offsets,
                                          scalar_to_block_map, matrix);
  }

  template <typename I>
  void update_csc_values(Graph<T, S> *graph, CSCMatrix<S, I> &matrix) {
    (void)graph;
    csc::update_scalar_csc_values<S, I>(
        pose_dim, values, d_col_pointers, d_row_indices, d_offsets,
        d_schur_offsets, scalar_to_block_map, matrix);
  }

  void compute_landmark_update(Graph<T, S> *graph, StreamPool &streams, T *xl,
                               T *xp) {
    auto stream = streams.select(0);
    const auto &offsets = graph->get_offset_vector();
    if (landmark_col_start >= num_block_columns) {
      return;
    }

    T *b = graph->get_b().data().get();

    // tmp_l = Hpl^T * xp
    execute_HplT_vector_multiply(graph, streams, l_workspace.data().get(), xp);

    // rhs_l = b_l - tmp_l
    thrust::copy_n(thrust::cuda::par_nosync.on(stream), b + pose_dim,
                   landmark_dim, xl);
    ops::axpy_async(stream, landmark_dim, xl, static_cast<T>(-1),
                    l_workspace.data().get(), xl);

    // xl = Hll^(-1) * rhs_l
    execute_diagonal_inverse_multiply(graph, streams, xl, xl);

    cudaStreamSynchronize(stream);
  }

private:
  // Computes the structure of S = Hpp - Hpl*Hll^(-1)*Hpl^T
  void build_schur_structure(Graph<T, S> *graph, StreamPool &streams) {
    size_t num_values = 0;
    block_indices.clear();
    // 1. To compute the Schur matrix, we need to know which blocks are
    // filled-in

    // 1.1 First we include all the blocks in Hpp (everything less than the
    // lowest eliminated block column)
    for (size_t col = 0; col < landmark_col_start; col++) {
      const size_t col_start = h_block_col_pointers[col];
      const size_t col_end = h_block_col_pointers[col + 1];
      for (size_t ka = col_start; ka < col_end; ka++) {
        block_indices.emplace(BlockCoordinates{h_block_row_indices[ka], col},
                              0);
      }
    }

    // 1.2 Next figure out which blocks are filled in due to the operation
    // Hpl*Hll^(-1)*Hpl^T  using the upper triangular Hessian.

    if (landmark_col_start < num_block_columns) {
      const size_t num_landmark_cols = num_block_columns - landmark_col_start;
      d_pose_counts.resize(num_landmark_cols);
      d_pair_counts.resize(num_landmark_cols);
      d_pair_offsets.resize(num_landmark_cols + 1);

      constexpr size_t threads_per_block = 256;
      const size_t count_blocks =
          (num_landmark_cols + threads_per_block - 1) / threads_per_block;

      ops::count_pose_rows_per_landmark_column_kernel<<<count_blocks,
                                                        threads_per_block>>>(
          H.get_block_col_pointers().data().get(),
          H.get_block_row_indices().data().get(), landmark_col_start,
          num_block_columns, d_pose_counts.data().get());

      thrust::transform(thrust::device, d_pose_counts.begin(),
                        d_pose_counts.end(), d_pair_counts.begin(),
                        PairCountFromPoseCount{});

      thrust::exclusive_scan(thrust::device, d_pair_counts.begin(),
                             d_pair_counts.end(), d_pair_offsets.begin());

      const size_t total_pairs =
          thrust::reduce(thrust::device, d_pair_counts.begin(),
                         d_pair_counts.end(), size_t(0));
      d_pair_offsets[num_landmark_cols] = total_pairs;

      if (total_pairs > 0) {
        d_pairs.resize(total_pairs);
        ops::fill_schur_structure_pairs_kernel<<<count_blocks,
                                                 threads_per_block>>>(
            H.get_block_col_pointers().data().get(),
            H.get_block_row_indices().data().get(), landmark_col_start,
            num_block_columns, d_pose_counts.data().get(),
            d_pair_offsets.data().get(), d_pairs.data().get());

        thrust::sort(thrust::device, d_pairs.begin(), d_pairs.end(),
                     BlockCoordColMajorLess{});

        auto unique_end =
            thrust::unique(thrust::device, d_pairs.begin(), d_pairs.end());
        d_pairs.resize(unique_end - d_pairs.begin());

        h_pairs = d_pairs;
        for (const auto &coord : h_pairs) {
          block_indices.emplace(coord, 0);
        }
      }
    }

    // Count scalar values and assign offsets after the sparsity pattern is set.
    for (auto &entry : block_indices) {
      entry.second = num_values;
      num_values += graph->get_variable_dimension(entry.first.row) *
                    graph->get_variable_dimension(entry.first.col);
    }

    values.resize(num_values);
  }

  // Sets up computation for Schur += Hpl*Hll^(-1)*Hpl^T.
  // Assumes this is called after preparing structure of  Hll^(-1) .
  // The inner dimension (the dimension of an l-block) serves as the loop bound.
  // We store one MulOp buffer per unique block dimension combination. The other
  // dimensions can be set dynamically at runtime. Only upper triangular blocks
  // are computed.
  void setup_schur_multiplication(Graph<T, S> *graph, StreamPool &streams) {
    auto stream = streams.select(0);

    for (auto &it : mul_ops) {
      it.second.clear();
    }

    for (auto &it : h_mul_ops) {
      it.second.clear();
    }

    if (landmark_col_start >= num_block_columns) {
      return;
    }

    block_dims.resize(num_block_columns);
    for (size_t b = 0; b < num_block_columns; b++) {
      block_dims[b] = graph->get_variable_dimension(b);
    }

    S *s_values = values.data().get();
    S *h_values = H.get_values_ptr();
    S *diag_values = diagonal_values.data().get();

    const size_t num_landmark_cols = num_block_columns - landmark_col_start;
    d_pose_counts.resize(num_landmark_cols);
    d_pair_counts.resize(num_landmark_cols);
    d_pair_offsets.resize(num_landmark_cols + 1);

    constexpr size_t threads_per_block = 256;
    const size_t count_blocks =
        (num_landmark_cols + threads_per_block - 1) / threads_per_block;

    ops::count_pose_rows_per_landmark_column_kernel<<<
        count_blocks, threads_per_block, 0, stream>>>(
        H.get_block_col_pointers().data().get(),
        H.get_block_row_indices().data().get(), landmark_col_start,
        num_block_columns, d_pose_counts.data().get());

    thrust::transform(thrust::cuda::par_nosync.on(stream),
                      d_pose_counts.begin(), d_pose_counts.end(),
                      d_pair_counts.begin(), PairCountFromPoseCount{});

    thrust::exclusive_scan(thrust::cuda::par_nosync.on(stream),
                           d_pair_counts.begin(), d_pair_counts.end(),
                           d_pair_offsets.begin());

    const size_t total_pairs =
        thrust::reduce(thrust::cuda::par_nosync.on(stream),
                       d_pair_counts.begin(), d_pair_counts.end(), size_t(0));

    if (total_pairs == 0) {
      d_mul_tuples.clear();
      h_mul_tuples.clear();
      return;
    }

    d_mul_tuples.resize(total_pairs);

    ops::fill_schur_mul_tuples_kernel<<<count_blocks, threads_per_block, 0,
                                        stream>>>(
        H.get_block_col_pointers().data().get(),
        H.get_block_row_indices().data().get(),
        H.get_block_value_offsets().data().get(), landmark_col_start,
        num_block_columns, d_pose_counts.data().get(),
        d_pair_offsets.data().get(), d_mul_tuples.data().get());

    cudaStreamSynchronize(stream);

    h_mul_tuples = d_mul_tuples;

    // Build MulOp groups on host from GPU-generated tuples.
    for (const auto &tuple : h_mul_tuples) {
      const size_t l = tuple.landmark_col;
      const BlockCoordinates ll{l, l};
      const auto mid_it = diagonal_indices.find(ll);
      if (mid_it == diagonal_indices.end()) {
        continue;
      }

      const size_t dim_l = block_dims[l];
      const size_t i = tuple.pose_row_i;
      const size_t j = tuple.pose_row_j;
      const BlockCoordinates dst{i, j};
      const auto dst_it = block_indices.find(dst);
      if (dst_it == block_indices.end()) {
        continue;
      }

      const ProductDim pd{block_dims[i], dim_l, block_dims[j]};
      h_mul_ops[pd].push_back(ops::MulOp<S>{
          s_values + dst_it->second,
          h_values + tuple.left_offset,
          diag_values + mid_it->second,
          h_values + tuple.right_offset,
      });
    }

    for (auto &entry : h_mul_ops) {
      mul_ops[entry.first] = entry.second;
    }
  }

  void execute_Hpp_copy(Graph<T, S> *graph, StreamPool &streams) {
    (void)graph;
    auto stream = streams.select(0);
    thrust::fill(thrust::cuda::par_nosync.on(stream), values.begin(),
                 values.end(), static_cast<S>(0.0));

    const S *h_values = H.get_values_ptr();
    S *s_values = values.data().get();

    constexpr size_t threads_per_block = 256;
    for (const auto &entry : hpp_copy_ops) {
      const auto &dim = entry.first;
      const auto &ops_dev = entry.second;
      const size_t num_ops = ops_dev.size();
      if (num_ops == 0) {
        continue;
      }

      const size_t total = num_ops * dim.row * dim.col;
      const size_t blocks = (total + threads_per_block - 1) / threads_per_block;
      ops::block_copy_batched_kernel<S>
          <<<blocks, threads_per_block, 0, stream>>>(h_values, s_values,
                                                     ops_dev.data().get(),
                                                     num_ops, dim.row, dim.col);
    }

    cudaStreamSynchronize(stream);
  }

  void setup_Hpp_copy(Graph<T, S> *graph, StreamPool &streams) {
    (void)streams;
    for (auto &it : hpp_copy_ops) {
      it.second.clear();
    }
    for (auto &it : h_hpp_copy_ops) {
      it.second.clear();
    }

    for (size_t col = 0; col < landmark_col_start; col++) {
      const size_t col_start = h_block_col_pointers[col];
      const size_t col_end = h_block_col_pointers[col + 1];
      for (size_t ka = col_start; ka < col_end; ka++) {
        const size_t row = h_block_row_indices[ka];
        const BlockCoordinates coord{row, col};

        const auto out_it = block_indices.find(coord);
        if (out_it == block_indices.end()) {
          continue;
        }

        const MatVecDim dim{graph->get_variable_dimension(row),
                            graph->get_variable_dimension(col)};
        h_hpp_copy_ops[dim].push_back(
            ops::BlockCopyOp{h_block_offsets[ka], out_it->second});
      }
    }

    for (auto &entry : h_hpp_copy_ops) {
      hpp_copy_ops[entry.first] = entry.second;
    }
  }

  void execute_schur_multiplication(Graph<T, S> *graph, StreamPool &streams) {
    const auto stream = streams.select(0);

    constexpr size_t threads_per_block = 256;
    for (auto &entry : mul_ops) {
      const auto &pd = entry.first;
      auto &ops_dev = entry.second;
      const size_t num_ops = ops_dev.size();

      if (num_ops > 0) {
        const size_t num_threads = num_ops * pd.dim_a * pd.dim_c;
        const size_t num_blocks =
            (num_threads + threads_per_block - 1) / threads_per_block;

        // auto stream = streams.select(stream_idx++);

        ops::schur_block_product_kernel<T, S>
            <<<num_blocks, threads_per_block, 0, stream>>>(
                ops_dev.data().get(), num_ops, pd.dim_a, pd.dim_b, pd.dim_c);
      }
    }
    cudaStreamSynchronize(stream);
    // streams.sync_all();
  }

  /*
    Sets up an operation for Hll^(-1) multiplying a landmark-sized vector.
    We need to do this twice: once for computing b_Schur and another time for
    computing the landmark update dx_l.
  */
  void setup_diagonal_inverse_multiply(Graph<T, S> *graph,
                                       StreamPool &streams) {
    (void)streams;
    const auto &offsets = graph->get_offset_vector();
    diagonal_value_offsets_by_dim.clear();
    landmark_local_offsets_by_dim.clear();

    if (landmark_col_start >= offsets.size() - 1) {
      l_workspace.clear();
      return;
    }

    l_workspace.resize(landmark_dim);
    diagonal_workspace.resize(landmark_dim);

    for (const auto &entry : diagonal_coords_by_dim) {
      const size_t dim = entry.first;
      const auto &coords = entry.second;

      h_diagonal_offsets.resize(coords.size());
      h_local_offsets.resize(coords.size());

      for (size_t i = 0; i < coords.size(); i++) {
        const auto &coord = coords[i];
        h_diagonal_offsets[i] = diagonal_indices.at(coord);
        h_local_offsets[i] = offsets[coord.col] - pose_dim;
      }

      diagonal_value_offsets_by_dim[dim] = h_diagonal_offsets;
      landmark_local_offsets_by_dim[dim] = h_local_offsets;
    }
  }

  void execute_diagonal_inverse_multiply(Graph<T, S> *graph,
                                         StreamPool &streams, T *vec_out,
                                         T *vec_in) {
    auto stream = streams.select(0);

    constexpr size_t threads_per_block = 256;
    const P *diag_values_ptr = diagonal_values.data().get();
    T *kernel_out = vec_out;
    if (vec_out == vec_in) {
      kernel_out = diagonal_workspace.data().get();
    }

    for (const auto &entry : diagonal_value_offsets_by_dim) {
      const size_t dim = entry.first;
      const auto &a_offsets_dev = entry.second;
      const auto vec_it = landmark_local_offsets_by_dim.find(dim);
      if (vec_it == landmark_local_offsets_by_dim.end()) {
        continue;
      }
      const auto &vec_offsets_dev = vec_it->second;
      const size_t num_blocks = a_offsets_dev.size();
      if (num_blocks == 0) {
        continue;
      }

      const size_t total_rows = num_blocks * dim;
      const size_t blocks =
          (total_rows + threads_per_block - 1) / threads_per_block;

      ops::block_matvec_assign_batched_kernel<T, P, T>
          <<<blocks, threads_per_block, 0, stream>>>(
              diag_values_ptr, a_offsets_dev.data().get(), vec_in, kernel_out,
              vec_offsets_dev.data().get(), num_blocks, dim);
    }

    if (vec_out == vec_in) {
      thrust::copy_n(thrust::cuda::par_nosync.on(stream), kernel_out,
                     landmark_dim, vec_out);
    }

    cudaStreamSynchronize(stream);
  }

  void setup_Hpl_vector_multiply(Graph<T, S> *graph, StreamPool &streams) {
    (void)streams;
    const auto &offsets = graph->get_offset_vector();
    p_workspace.resize(pose_dim);

    for (auto &it : hpl_vec_ops) {
      it.second.clear();
    }
    for (auto &it : h_hpl_vec_ops) {
      it.second.clear();
    }

    if (landmark_col_start >= num_block_columns) {
      return;
    }

    for (size_t l = landmark_col_start; l < num_block_columns; l++) {
      const size_t col_start = h_block_col_pointers[l];
      const size_t col_end = h_block_col_pointers[l + 1];
      const size_t dim_l = graph->get_variable_dimension(l);
      const size_t x_offset = offsets[l] - pose_dim;

      for (size_t ka = col_start; ka < col_end; ka++) {
        const size_t i = h_block_row_indices[ka];
        if (i >= landmark_col_start) {
          break;
        }

        const size_t dim_i = graph->get_variable_dimension(i);
        const MatVecDim mvd{dim_i, dim_l};
        h_hpl_vec_ops[mvd].push_back(
            ops::HplMatVecOp{h_block_offsets[ka], x_offset, offsets[i]});
      }
    }

    for (auto &entry : h_hpl_vec_ops) {
      hpl_vec_ops[entry.first] = entry.second;
    }
  }

  void execute_Hpl_vector_multiply(Graph<T, S> *graph, StreamPool &streams,
                                   T *vec_out, T *vec_in) {
    auto stream = streams.select(0);
    const auto &offsets = graph->get_offset_vector();

    thrust::fill(thrust::cuda::par.on(stream), vec_out, vec_out + pose_dim,
                 static_cast<T>(0));

    const S *h_values = H.get_values_ptr();

    constexpr size_t threads_per_block = 256;
    for (const auto &entry : hpl_vec_ops) {
      const auto &mvd = entry.first;
      const auto &ops_dev = entry.second;
      const size_t num_ops = ops_dev.size();
      if (num_ops > 0) {
        const size_t total_rows = num_ops * mvd.row;
        const size_t blocks =
            (total_rows + threads_per_block - 1) / threads_per_block;
        ops::block_matvec_add_batched_kernel<T, S>
            <<<blocks, threads_per_block, 0, stream>>>(
                h_values, ops_dev.data().get(), num_ops, vec_in, vec_out,
                mvd.row, mvd.col);
      }
    }

    cudaStreamSynchronize(stream);
  }

  void setup_b_Schur_computation(Graph<T, S> *graph, StreamPool &streams) {
    (void)streams;
    const auto &offsets = graph->get_offset_vector();
    if (landmark_col_start >= offsets.size()) {
      b_Schur.clear();
      p_workspace.clear();
      l_workspace.clear();
      return;
    }

    b_Schur.resize(pose_dim);
    p_workspace.resize(pose_dim);
    l_workspace.resize(landmark_dim);
  }

  void execute_b_Schur_computation(Graph<T, S> *graph, StreamPool &streams,
                                   T *b_Schur) {
    auto stream = streams.select(0);
    const auto &offsets = graph->get_offset_vector();

    T *b = graph->get_b().data().get();

    thrust::copy_n(thrust::cuda::par.on(stream), b, pose_dim, b_Schur);

    execute_diagonal_inverse_multiply(graph, streams, l_workspace.data().get(),
                                      b + pose_dim);

    execute_Hpl_vector_multiply(graph, streams, p_workspace.data().get(),
                                l_workspace.data().get());

    ops::axpy_async(stream, pose_dim, b_Schur, static_cast<T>(-1.0),
                    p_workspace.data().get(), b_Schur);

    cudaStreamSynchronize(stream);
  }

  void setup_HplT_vector_multiply(Graph<T, S> *graph, StreamPool &streams) {
    (void)streams;
    const auto &offsets = graph->get_offset_vector();
    for (auto &it : hplt_vec_ops) {
      it.second.clear();
    }
    for (auto &it : h_hplt_vec_ops) {
      it.second.clear();
    }

    if (landmark_col_start >= offsets.size() - 1) {
      l_workspace.clear();
      return;
    }
    l_workspace.resize(landmark_dim);

    if (landmark_col_start >= num_block_columns) {
      return;
    }

    for (size_t l = landmark_col_start; l < num_block_columns; l++) {
      const size_t col_start = h_block_col_pointers[l];
      const size_t col_end = h_block_col_pointers[l + 1];
      const size_t dim_l = graph->get_variable_dimension(l);
      const size_t y_offset = offsets[l] - pose_dim;

      for (size_t ka = col_start; ka < col_end; ka++) {
        const size_t i = h_block_row_indices[ka];
        if (i >= landmark_col_start) {
          break;
        }

        const size_t dim_i = graph->get_variable_dimension(i);
        const MatVecDim mvd{dim_i, dim_l};
        h_hplt_vec_ops[mvd].push_back(
            ops::HplMatVecOp{h_block_offsets[ka], offsets[i], y_offset});
      }
    }

    for (auto &entry : h_hplt_vec_ops) {
      hplt_vec_ops[entry.first] = entry.second;
    }
  }

  void execute_HplT_vector_multiply(Graph<T, S> *graph, StreamPool &streams,
                                    T *vec_out, T *vec_in) {
    auto stream = streams.select(0);
    const auto &offsets = graph->get_offset_vector();
    if (landmark_col_start >= num_block_columns) {
      return;
    }

    thrust::fill(thrust::cuda::par_nosync.on(stream), vec_out,
                 vec_out + landmark_dim, static_cast<T>(0));

    const S *h_values = H.get_values_ptr();

    constexpr size_t threads_per_block = 256;
    for (const auto &entry : hplt_vec_ops) {
      const auto &mvd = entry.first;
      const auto &ops_dev = entry.second;
      const size_t num_ops = ops_dev.size();
      if (num_ops == 0) {
        continue;
      }

      const size_t total_cols = num_ops * mvd.col;
      const size_t blocks =
          (total_cols + threads_per_block - 1) / threads_per_block;
      ops::block_matvec_transpose_add_batched_kernel<T, S>
          <<<blocks, threads_per_block, 0, stream>>>(
              h_values, ops_dev.data().get(), num_ops, vec_in, vec_out, mvd.row,
              mvd.col);
    }

    cudaStreamSynchronize(stream);
  }

  // Builds the structure of Hll^(-1)
  void build_diagonal_structure(Graph<T, S> *graph, StreamPool &streams) {
    diagonal_indices.clear();
    diagonal_coords_by_dim.clear();

    // Keep only l-diagonal blocks from Hessian structure.
    size_t diagonal_num_values = 0;
    for (size_t block_col = landmark_col_start; block_col < num_block_columns;
         block_col++) {
      const BlockCoordinates coord{block_col, block_col};
      diagonal_indices.emplace(coord, diagonal_num_values);
      const size_t dim = graph->get_variable_dimension(block_col);
      diagonal_coords_by_dim[dim].push_back(coord);
      diagonal_num_values += dim * dim;
    }

    diagonal_values.resize(diagonal_num_values);
    setup_batched_inverse_buffers(streams);
  }

  void setup_batched_inverse_buffers(StreamPool &streams) {
    auto stream = streams.select(0);
    P *diag_values = diagonal_values.data().get();
    const P *h_values = H.get_values_ptr();

    A_ptrs.clear();
    Ainv_ptrs.clear();
    A_ptrs_device_by_dim.clear();
    Ainv_ptrs_device_by_dim.clear();
    info_by_dim.clear();

    for (auto &entry : diagonal_coords_by_dim) {
      const size_t dim = entry.first;
      auto &coords = entry.second;
      const size_t num_blocks = coords.size();

      auto &A_ptrs_device = A_ptrs_device_by_dim[dim];
      auto &Ainv_ptrs_device = Ainv_ptrs_device_by_dim[dim];
      auto &info = info_by_dim[dim];

      A_ptrs.resize(num_blocks);
      Ainv_ptrs.resize(num_blocks);
      A_ptrs_device.resize(num_blocks);
      Ainv_ptrs_device.resize(num_blocks);
      info.resize(num_blocks);

      for (size_t i = 0; i < num_blocks; i++) {
        const auto &coord = coords[i];
        const size_t src_offset = H.block_indices.at(coord);
        const size_t dst_offset = diagonal_indices.at(coord);
        this->A_ptrs[i] = const_cast<P *>(h_values + src_offset);
        this->Ainv_ptrs[i] = diag_values + dst_offset;
      }

      cudaMemcpyAsync(A_ptrs_device.data().get(), this->A_ptrs.data(),
                      sizeof(P *) * num_blocks, cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(Ainv_ptrs_device.data().get(), this->Ainv_ptrs.data(),
                      sizeof(P *) * num_blocks, cudaMemcpyHostToDevice, stream);

      cudaStreamSynchronize(stream);

      // Clear for reuse
      A_ptrs.clear();
      Ainv_ptrs.clear();
    }
  }

  void execute_block_diagonal_inversion(Graph<T, S> *graph,
                                        StreamPool &streams) {
    (void)graph;
    auto stream = streams.select(0);
    cublasSetStream(handle, stream);

    // cublas<t>matinvBatched only supports small matrices.
    constexpr size_t max_supported_dim = 32;

    for (auto &entry : diagonal_coords_by_dim) {
      const size_t block_dim = entry.first;
      if (block_dim > max_supported_dim) {
        std::cerr << "Runtime error: Schur diagonal inversion received block "
                     "dimension "
                  << block_dim
                  << ", but cublas matinvBatched supports at "
                     "most "
                  << max_supported_dim << "." << std::endl;
        break;
      }

      auto &A_ptrs_device = A_ptrs_device_by_dim[block_dim];
      auto &Ainv_ptrs_device = Ainv_ptrs_device_by_dim[block_dim];
      auto &info = info_by_dim[block_dim];
      const int num_blocks = static_cast<int>(entry.second.size());

      // if (ops::launch_small_block_inverse_batched<P>(
      //         block_dim, A_ptrs_device.data().get(),
      //         Ainv_ptrs_device.data().get(), static_cast<size_t>(num_blocks),
      //         stream)) {
      //   continue;
      // }

      if constexpr (std::is_same<P, double>::value) {
        cublasDmatinvBatched(
            handle, static_cast<int>(block_dim), A_ptrs_device.data().get(),
            static_cast<int>(block_dim), Ainv_ptrs_device.data().get(),
            static_cast<int>(block_dim), info.data().get(), num_blocks);
      } else if constexpr (std::is_same<P, float>::value) {
        cublasSmatinvBatched(
            handle, static_cast<int>(block_dim), A_ptrs_device.data().get(),
            static_cast<int>(block_dim), Ainv_ptrs_device.data().get(),
            static_cast<int>(block_dim), info.data().get(), num_blocks);
      }
    }

    cudaStreamSynchronize(stream);
  }
};

} // namespace graphite
