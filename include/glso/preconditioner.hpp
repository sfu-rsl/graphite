#pragma once
#include <Eigen/Dense>
#include <cublas_v2.h>
#include <glso/factor.hpp>
#include <glso/op.hpp>
#include <glso/vertex.hpp>
#include <thrust/execution_policy.h>

namespace glso {

template <typename T, typename S> class Preconditioner {
public:
  virtual void
  precompute(GraphVisitor<T, S> &visitor,
             std::vector<BaseVertexDescriptor<T, S> *> &vertex_descriptors,
             std::vector<BaseFactorDescriptor<T, S> *> &factor_descriptors,
             size_t dimension, S mu) = 0;

  virtual void apply(GraphVisitor<T, S> &visitor, S *z, const S *r) = 0;
};

template <typename T, typename S>
class IdentityPreconditioner : public Preconditioner<T, S> {
private:
  size_t dimension;

public:
  void precompute(GraphVisitor<T, S> &visitor,
                  std::vector<BaseVertexDescriptor<T, S> *> &vertex_descriptors,
                  std::vector<BaseFactorDescriptor<T, S> *> &factor_descriptors,
                  size_t dimension, S mu) override {
    this->dimension = dimension;
  }

  void apply(GraphVisitor<T, S> &visitor, S *z, const S *r) override {
    cudaMemcpy(z, r, dimension * sizeof(S), cudaMemcpyDeviceToDevice);
  }
};

template <typename T, typename S>
class BlockJacobiPreconditioner : public Preconditioner<T, S> {
private:
  using P = std::conditional_t<std::is_same<S, ghalf>::value, T, S>;
  size_t dimension;
  std::vector<std::pair<size_t, size_t>> block_sizes;
  std::unordered_map<BaseVertexDescriptor<T, S> *, thrust::device_vector<S>>
      block_diagonals;

  std::unordered_map<BaseVertexDescriptor<T, S> *, thrust::device_vector<P>>
      hp_diagonals; // higher precision diagonals for inversion
  std::vector<BaseVertexDescriptor<T, S> *> *vds;
  cublasHandle_t handle;

  // For batched inversion
  // TODO: Figure out a better way to handle the memory
  thrust::device_vector<P> Ainv_data;
  thrust::host_vector<P *> A_ptrs, Ainv_ptrs;
  thrust::device_vector<P *> A_ptrs_device, Ainv_ptrs_device;
  thrust::device_vector<int> info;

public:
  BlockJacobiPreconditioner() {
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  }

  ~BlockJacobiPreconditioner() { cublasDestroy(handle); }

  void precompute(GraphVisitor<T, S> &visitor,
                  std::vector<BaseVertexDescriptor<T, S> *> &vertex_descriptors,
                  std::vector<BaseFactorDescriptor<T, S> *> &factor_descriptors,
                  size_t dimension, S mu) override {
    this->dimension = dimension;
    this->vds = &vertex_descriptors;

    for (auto &desc : vertex_descriptors) {
      // Reserve space
      const auto d = desc->dimension();
      const size_t num_values =
          d * d * desc->count(); // this is not tightly packed since count
                                 // includes fixed vertices
      block_diagonals[desc] =
          thrust::device_vector<S>(num_values, static_cast<S>(0));
      // block_diagonals.insert(desc, thrust::device_vector<T>(num_values, 0));
      if constexpr (std::is_same<S, ghalf>::value) {
        hp_diagonals[desc].resize(num_values);
      }
    }

    // Compute Hessian blocks on the diagonal
    for (auto &desc : vertex_descriptors) {
      if constexpr (std::is_same<S, ghalf>::value) {
        thrust::fill(hp_diagonals[desc].begin(), hp_diagonals[desc].end(),
                    static_cast<P>(0));
      }
      else {
        thrust::fill(block_diagonals[desc].begin(), block_diagonals[desc].end(),
                    static_cast<S>(0));
      }
    }
    for (auto &desc : factor_descriptors) {
      if constexpr (std::is_same<S, ghalf>::value) {
        desc->visit_block_diagonal(visitor, hp_diagonals);
      }
      else {
        desc->visit_block_diagonal(visitor, block_diagonals);
      }
    }
    cudaDeviceSynchronize();

    // Invert the blocks

    for (auto &desc : vertex_descriptors) {
      if constexpr (std::is_same<S, ghalf>::value) {
        desc->visit_augment_block_diagonal(
            visitor, hp_diagonals[desc].data().get(), mu);
      }
      else {
        desc->visit_augment_block_diagonal(
            visitor, block_diagonals[desc].data().get(), mu);
      }
      // Invert the block diagonal using cublas
      const auto d = desc->dimension();
      const size_t num_blocks = desc->count();
      const auto block_size = d * d;

      A_ptrs.resize(num_blocks);
      Ainv_ptrs.resize(num_blocks);
      Ainv_data.resize(num_blocks * block_size);
      info.resize(num_blocks);

      // P *a_ptr = block_diagonals[desc].data().get();

      P *a_ptr = nullptr;

      if constexpr (std::is_same<S, ghalf>::value) {
        // hp_diagonals[desc].resize(num_blocks * block_size);
        // // Copy to higher precision
        // thrust::transform(thrust::device, block_diagonals[desc].begin(),
        //                   block_diagonals[desc].end(),
        //                   hp_diagonals[desc].begin(),
        //                   [] __device__(S val) { return static_cast<P>(val); });

        a_ptr = hp_diagonals[desc].data().get();
      } else {
        a_ptr = block_diagonals[desc].data().get();
      }

      P *a_inv_ptr = Ainv_data.data().get();
      for (size_t i = 0; i < num_blocks; ++i) {
        A_ptrs[i] = a_ptr + i * block_size;
        Ainv_ptrs[i] = a_inv_ptr + i * block_size;
      }

      A_ptrs_device = A_ptrs;
      Ainv_ptrs_device = Ainv_ptrs;

      if constexpr (std::is_same<P, double>::value) {

        cublasDmatinvBatched(handle, d, A_ptrs_device.data().get(), d,
                             Ainv_ptrs_device.data().get(), d,
                             info.data().get(), num_blocks);
      } else if constexpr (std::is_same<P, float>::value) {
        // std::cout << "Inverting block diagonal with float precision." << std::endl;
        // thrust::fill(hp_diagonals[desc].begin(),
        //              hp_diagonals[desc].end(), static_cast<P>(0));
        cublasSmatinvBatched(handle, d, A_ptrs_device.data().get(), d,
                             Ainv_ptrs_device.data().get(), d,
                             info.data().get(), num_blocks);
      } else {
        static_assert(std::is_same<S, ghalf>::value ||
                          std::is_same<S, float>::value ||
                          std::is_same<S, double>::value,
                      "BlockJacobiPreconditioner only supports ghalf, float, or "
                      "double types.");
      }

      cudaDeviceSynchronize();

      // Check for errors in inversion
      // thrust::host_vector<int> info_host = info;
      // for (size_t i = 0; i < num_blocks; ++i) {
      //   if (info_host[i] != 0) {
      //     std::cerr << "Error in matrix inversion for block " << i
      //               << ": info = " << info_host[i] << std::endl;
      //   }
      // }

      // Copy back
      if constexpr (std::is_same<S, ghalf>::value) {
        thrust::transform(thrust::device, Ainv_data.begin(), Ainv_data.end(),
                          block_diagonals[desc].begin(),
                          [] __device__(P val) { return static_cast<S>(val); });
      } else {
        thrust::copy(thrust::device, Ainv_data.begin(), Ainv_data.end(),
                     block_diagonals[desc].begin());
      }
    }

    cudaDeviceSynchronize();
  }

  void apply(GraphVisitor<T, S> &visitor, S *z, const S *r) override {
    // Apply the preconditioner
    for (auto &desc : *vds) {
      const auto d = desc->dimension();
      S *blocks = block_diagonals[desc].data().get();
      // std::cout << "bd size: " << block_diagonals[desc].size() << std::endl;
      desc->visit_apply_block_jacobi(visitor, z, r, blocks);
    }
    cudaDeviceSynchronize();
  }
};

} // namespace glso