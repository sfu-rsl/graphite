/// @file schur.hpp
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <graphite/block.hpp>
#include <graphite/ops/common.hpp>

namespace graphite {

namespace ops {
/**
 * @brief Stores offsets for Hpl*Hll^(-1)*Hpl^T operation.
 */
template <typename T> struct MulOp {
  T *destination;
  T *left;
  T *middle;
  T *right;
};

struct HplMatVecOp {
  size_t a_offset;
  size_t x_offset;
  size_t y_offset;
};

struct BlockCopyOp {
  size_t src_offset;
  size_t dst_offset;
};

struct SchurMulTuple {
  size_t landmark_col;
  size_t pose_row_i;
  size_t pose_row_j;
  size_t left_offset;
  size_t right_offset;
};

__global__ void count_pose_rows_per_landmark_column_kernel(
    const size_t *col_pointers, const size_t *row_indices,
    size_t landmark_col_start, size_t num_block_columns, size_t *pose_counts) {
  const size_t idx = get_thread_id();
  const size_t num_landmark_cols = num_block_columns - landmark_col_start;
  if (idx >= num_landmark_cols) {
    return;
  }

  const size_t l = landmark_col_start + idx;
  const size_t col_start = col_pointers[l];
  const size_t col_end = col_pointers[l + 1];

  size_t count = 0;
  for (size_t ka = col_start; ka < col_end; ka++) {
    if (row_indices[ka] >= landmark_col_start) {
      break;
    }
    count++;
  }
  pose_counts[idx] = count;
}

__global__ void fill_schur_structure_pairs_kernel(const size_t *col_pointers,
                                                  const size_t *row_indices,
                                                  size_t landmark_col_start,
                                                  size_t num_block_columns,
                                                  const size_t *pose_counts,
                                                  const size_t *pair_offsets,
                                                  BlockCoordinates *pairs_out) {
  const size_t idx = get_thread_id();
  const size_t num_landmark_cols = num_block_columns - landmark_col_start;
  if (idx >= num_landmark_cols) {
    return;
  }

  const size_t l = landmark_col_start + idx;
  const size_t col_start = col_pointers[l];
  const size_t pose_count = pose_counts[idx];
  size_t out_offset = pair_offsets[idx];

  for (size_t a = 0; a < pose_count; a++) {
    const size_t i = row_indices[col_start + a];
    for (size_t b = a; b < pose_count; b++) {
      const size_t j = row_indices[col_start + b];
      pairs_out[out_offset++] = BlockCoordinates{i, j};
    }
  }
}

__global__ void fill_schur_mul_tuples_kernel(
    const size_t *col_pointers, const size_t *row_indices,
    const size_t *block_offsets, size_t landmark_col_start,
    size_t num_block_columns, const size_t *pose_counts,
    const size_t *pair_offsets, SchurMulTuple *tuples_out) {
  const size_t idx = get_thread_id();
  const size_t num_landmark_cols = num_block_columns - landmark_col_start;
  if (idx >= num_landmark_cols) {
    return;
  }

  const size_t l = landmark_col_start + idx;
  const size_t col_start = col_pointers[l];
  const size_t pose_count = pose_counts[idx];
  size_t out_offset = pair_offsets[idx];

  for (size_t a = 0; a < pose_count; a++) {
    const size_t ka = col_start + a;
    const size_t i = row_indices[ka];
    const size_t left_offset = block_offsets[ka];
    for (size_t b = a; b < pose_count; b++) {
      const size_t kb = col_start + b;
      const size_t j = row_indices[kb];
      tuples_out[out_offset++] =
          SchurMulTuple{l, i, j, left_offset, block_offsets[kb]};
    }
  }
}

template <typename T, typename S>
__global__ void schur_block_product_kernel(const MulOp<S> *ops, size_t num_ops,
                                           size_t dim_a, size_t dim_b,
                                           size_t dim_c) {
  const size_t idx = get_thread_id();
  const size_t block_size = dim_a * dim_c;
  const size_t op_id = idx / block_size;
  if (op_id >= num_ops) {
    return;
  }

  const size_t offset = idx % block_size;
  const size_t row = offset % dim_a;
  const size_t col = offset / dim_a;

  const auto &op = ops[op_id];
  const S *left = op.left;
  const S *middle = op.middle;
  const S *right = op.right;

  // Computes destination -= left * middle * right^T.
  T value = 0;
  for (size_t k = 0; k < dim_b; k++) {
    T m_rt = 0;
    for (size_t j = 0; j < dim_b; j++) {
      m_rt += static_cast<T>(middle[k + j * dim_b]) *
              static_cast<T>(right[col + j * dim_c]);
    }
    value += static_cast<T>(left[row + k * dim_a]) * m_rt;
  }

  atomicAdd(op.destination + (row + col * dim_a), static_cast<S>(-value));
}

template <typename highp, typename S, typename T>
__global__ void block_matvec_assign_batched_kernel(
    const S *values, const size_t *a_offsets, const T *x_base, T *y_base,
    const size_t *vec_offsets, size_t num_blocks, size_t dim) {
  const size_t idx = get_thread_id();
  const size_t total_rows = num_blocks * dim;
  if (idx >= total_rows) {
    return;
  }

  const size_t block_id = idx / dim;
  const size_t row = idx % dim;

  const S *A = values + a_offsets[block_id];
  const size_t vec_offset = vec_offsets[block_id];
  const T *x = x_base + vec_offset;
  T *y = y_base + vec_offset;

  T sum = 0;
  for (size_t c = 0; c < dim; c++) {
    sum += static_cast<T>(A[row + c * dim]) * static_cast<T>(x[c]);
  }
  y[row] = sum;
}

template <typename T, typename S>
__global__ void block_matvec_add_batched_kernel(
    const S *values, const HplMatVecOp *ops, const size_t num_ops,
    const T *x_base, T *y_base, const size_t rows, const size_t cols) {
  const size_t idx = get_thread_id();
  const size_t total_rows = num_ops * rows;
  if (idx >= total_rows) {
    return;
  }

  const size_t op_id = idx / rows;
  const size_t row = idx % rows;
  const auto &op = ops[op_id];

  const S *A = values + op.a_offset;
  const T *x = x_base + op.x_offset;
  T *y = y_base + op.y_offset;

  T sum = 0;
  for (size_t c = 0; c < cols; c++) {
    sum += static_cast<T>(A[row + c * rows]) * static_cast<T>(x[c]);
  }
  atomicAdd(y + row, sum);
}

template <typename T, typename S>
__global__ void block_matvec_transpose_add_batched_kernel(
    const S *values, const HplMatVecOp *ops, const size_t num_ops,
    const T *x_base, T *y_base, const size_t rows, const size_t cols) {
  const size_t idx = get_thread_id();
  const size_t total_cols = num_ops * cols;
  if (idx >= total_cols) {
    return;
  }

  const size_t op_id = idx / cols;
  const size_t col = idx % cols;
  const auto &op = ops[op_id];

  const S *A = values + op.a_offset;
  const T *x = x_base + op.x_offset;
  T *y = y_base + op.y_offset;

  T sum = 0;
  for (size_t r = 0; r < rows; r++) {
    sum += static_cast<T>(A[r + col * rows]) * static_cast<T>(x[r]);
  }
  atomicAdd(y + col, sum);
}

template <typename S>
__global__ void
block_copy_batched_kernel(const S *src_values, S *dst_values,
                          const BlockCopyOp *ops, const size_t num_ops,
                          const size_t rows, const size_t cols) {
  const size_t idx = get_thread_id();
  const size_t block_size = rows * cols;
  const size_t total = num_ops * block_size;
  if (idx >= total) {
    return;
  }

  const size_t op_id = idx / block_size;
  const size_t local_idx = idx % block_size;
  const auto &op = ops[op_id];
  dst_values[op.dst_offset + local_idx] = src_values[op.src_offset + local_idx];
}

/*
template <size_t dim, typename S>
__global__ void invert_small_block_ptrs_batched_kernel(S **A_ptrs,
                                                       S **Ainv_ptrs,
                                                       size_t num_blocks) {
  const size_t block_id = get_thread_id();
  if (block_id >= num_blocks) {
    return;
  }

  const S *A = A_ptrs[block_id];
  S *Ainv = Ainv_ptrs[block_id];
  using Block = Eigen::Matrix<S, dim, dim, Eigen::ColMajor>;
  Eigen::Map<const Block> A_map(A);
  Eigen::Map<Block> Ainv_map(Ainv);

  if constexpr (dim == 1) {
    Ainv_map(0, 0) = static_cast<S>(1) / A_map(0, 0);
  } else if constexpr (dim == 2 || dim == 3 || dim == 4) {
    Ainv_map = A_map.inverse();
  }
}

template <typename S>
bool launch_small_block_inverse_batched(size_t block_dim, S **A_ptrs_device,
                                        S **Ainv_ptrs_device, size_t num_blocks,
                                        cudaStream_t stream) {
  if (num_blocks == 0) {
    return true;
  }

  constexpr size_t threads_per_block = 128;
  const size_t blocks =
      (num_blocks + threads_per_block - 1) / threads_per_block;

  switch (block_dim) {
  case 1:
    invert_small_block_ptrs_batched_kernel<1, S>
        <<<blocks, threads_per_block, 0, stream>>>(
            A_ptrs_device, Ainv_ptrs_device, num_blocks);
    return true;
  case 2:
    invert_small_block_ptrs_batched_kernel<2, S>
        <<<blocks, threads_per_block, 0, stream>>>(
            A_ptrs_device, Ainv_ptrs_device, num_blocks);
    return true;
  case 3:
    invert_small_block_ptrs_batched_kernel<3, S>
        <<<blocks, threads_per_block, 0, stream>>>(
            A_ptrs_device, Ainv_ptrs_device, num_blocks);
    return true;
  case 4:
    invert_small_block_ptrs_batched_kernel<4, S>
        <<<blocks, threads_per_block, 0, stream>>>(
            A_ptrs_device, Ainv_ptrs_device, num_blocks);
    return true;
  default:
    return false;
  }
}
*/
} // namespace ops

} // namespace graphite