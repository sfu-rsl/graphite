/// @file block_jacobi_schur.hpp
#pragma once

#include <cublas_v2.h>
#include <graphite/ops/schur.hpp>
#include <graphite/preconditioner/schur_preconditioner.hpp>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <unordered_map>

namespace graphite {
/// @brief Block Jacobi preconditioner for PCG with the Schur complement
template <typename T, typename S>
class BlockJacobiSchurPreconditioner : public SchurPreconditioner<T, S> {
private:
  using P = std::conditional_t<is_low_precision<S>::value, T, S>;

  struct DimGroup {
    size_t dim;
    thrust::host_vector<size_t> h_src_offsets;
    thrust::host_vector<size_t> h_vec_offsets;
    thrust::host_vector<ops::BlockCopyOp> h_copy_ops;
    thrust::device_vector<size_t> d_a_offsets;
    thrust::device_vector<size_t> d_vec_offsets;
    thrust::device_vector<ops::BlockCopyOp> d_copy_ops;
    thrust::device_vector<P> blocks;
    thrust::device_vector<P> blocks_inv;
    thrust::host_vector<P *> h_A_ptrs;
    thrust::host_vector<P *> h_Ainv_ptrs;
    thrust::device_vector<P *> d_A_ptrs;
    thrust::device_vector<P *> d_Ainv_ptrs;
    thrust::device_vector<int> d_info;
  };

  std::unordered_map<size_t, DimGroup> dim_groups;
  size_t pose_dim;
  cublasHandle_t handle;

public:
  BlockJacobiSchurPreconditioner() : pose_dim(0) {
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  }

  ~BlockJacobiSchurPreconditioner() override { cublasDestroy(handle); }

  void update_structure(Graph<T, S> *graph, SchurComplement<T, S> *schur,
                        StreamPool &streams) override {
    (void)streams;
    dim_groups.clear();

    pose_dim = graph->get_offset_vector()[schur->lowest_eliminated_block_col];
    const auto &offsets = graph->get_offset_vector();

    for (size_t block = 0; block < schur->lowest_eliminated_block_col;
         ++block) {
      const size_t dim = graph->get_variable_dimension(block);
      auto diag_it = schur->block_indices.find(BlockCoordinates{block, block});
      if (diag_it == schur->block_indices.end()) {
        continue;
      }

      auto &group = dim_groups[dim];
      group.dim = dim;
      group.h_src_offsets.push_back(diag_it->second);
      group.h_vec_offsets.push_back(offsets[block]);
    }

    for (auto &entry : dim_groups) {
      auto &group = entry.second;
      const size_t num_blocks = group.h_src_offsets.size();
      const size_t block_size = group.dim * group.dim;

      group.blocks.resize(num_blocks * block_size);
      group.blocks_inv.resize(num_blocks * block_size);
      group.h_A_ptrs.resize(num_blocks);
      group.h_Ainv_ptrs.resize(num_blocks);
      group.d_A_ptrs.resize(num_blocks);
      group.d_Ainv_ptrs.resize(num_blocks);
      group.d_info.resize(num_blocks);
      group.d_vec_offsets = group.h_vec_offsets;
      group.h_copy_ops.resize(num_blocks);
      group.d_copy_ops.resize(num_blocks);

      thrust::host_vector<size_t> h_a_offsets(num_blocks);
      for (size_t i = 0; i < num_blocks; ++i) {
        h_a_offsets[i] = i * block_size;
        group.h_copy_ops[i] =
            ops::BlockCopyOp{group.h_src_offsets[i], h_a_offsets[i]};
      }
      group.d_a_offsets = h_a_offsets;
      group.d_copy_ops = group.h_copy_ops;

      P *blocks_ptr = group.blocks.data().get();
      P *blocks_inv_ptr = group.blocks_inv.data().get();
      for (size_t i = 0; i < num_blocks; ++i) {
        group.h_A_ptrs[i] = blocks_ptr + i * block_size;
        group.h_Ainv_ptrs[i] = blocks_inv_ptr + i * block_size;
      }

      cudaMemcpyAsync(group.d_A_ptrs.data().get(), group.h_A_ptrs.data(),
                      sizeof(P *) * num_blocks, cudaMemcpyHostToDevice,
                      streams.select(0));
      cudaMemcpyAsync(group.d_Ainv_ptrs.data().get(), group.h_Ainv_ptrs.data(),
                      sizeof(P *) * num_blocks, cudaMemcpyHostToDevice,
                      streams.select(0));
      cudaStreamSynchronize(streams.select(0));
    }
  }

  void update_values(Graph<T, S> *graph, SchurComplement<T, S> *schur,
                     StreamPool &streams) override {
    (void)graph;
    auto stream = streams.select(0);
    cublasSetStream(handle, stream);

    const S *schur_values = schur->values.data().get();
    constexpr size_t threads_per_block = 256;

    for (auto &entry : dim_groups) {
      auto &group = entry.second;
      const size_t num_blocks = group.h_src_offsets.size();
      if (num_blocks == 0) {
        continue;
      }
      const size_t block_size = group.dim * group.dim;
      P *blocks_ptr = group.blocks.data().get();

      const size_t total = num_blocks * block_size;
      const size_t blocks = (total + threads_per_block - 1) / threads_per_block;
      ops::block_copy_batched_kernel<S, P>
          <<<blocks, threads_per_block, 0, stream>>>(
              schur_values, blocks_ptr, group.d_copy_ops.data().get(),
              num_blocks, group.dim, group.dim);

      if constexpr (std::is_same<P, double>::value) {
        cublasDmatinvBatched(handle, group.dim, group.d_A_ptrs.data().get(),
                             group.dim, group.d_Ainv_ptrs.data().get(),
                             group.dim, group.d_info.data().get(), num_blocks);
      } else if constexpr (std::is_same<P, float>::value) {
        cublasSmatinvBatched(handle, group.dim, group.d_A_ptrs.data().get(),
                             group.dim, group.d_Ainv_ptrs.data().get(),
                             group.dim, group.d_info.data().get(), num_blocks);
      }
    }

    cudaStreamSynchronize(stream);
  }

  void set_damping_factor(Graph<T, S> *graph, SchurComplement<T, S> *schur,
                          T damping_factor, const bool use_identity,
                          StreamPool &streams) override {}

  void apply(Graph<T, S> *graph, SchurComplement<T, S> *schur, T *z, const T *r,
             StreamPool &streams) override {
    (void)graph;
    (void)schur;
    const auto stream = streams.select(0);
    thrust::fill(thrust::cuda::par_nosync.on(stream), z, z + pose_dim,
                 static_cast<T>(0));

    constexpr size_t threads_per_block = 256;
    for (auto &entry : dim_groups) {
      auto &group = entry.second;
      const size_t num_blocks = group.h_src_offsets.size();
      const size_t total_rows = num_blocks * group.dim;
      const size_t blocks =
          (total_rows + threads_per_block - 1) / threads_per_block;

      ops::block_matvec_assign_batched_kernel<T, P, T>
          <<<blocks, threads_per_block, 0, stream>>>(
              group.blocks_inv.data().get(), group.d_a_offsets.data().get(), r,
              z, group.d_vec_offsets.data().get(), num_blocks, group.dim);
    }
  }
};

} // namespace graphite
