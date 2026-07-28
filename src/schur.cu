#include <graphite/ops/schur.hpp>

namespace graphite {
namespace ops {

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

__global__ void fill_schur_structure_pairs_kernel(
    const size_t *col_pointers, const size_t *row_indices,
    const size_t landmark_col_start, const size_t num_block_columns,
    const size_t *pose_counts, const size_t *pair_offsets,
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

} // namespace ops
} // namespace graphite