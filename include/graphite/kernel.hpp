#pragma once
// #include <cublas_v2.h>
namespace graphite {
template <typename T>
__global__ void axpy_kernel(size_t n, T *z, const T a, const T *x, T *y) {
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
      static_cast<size_t>(threadIdx.x);
  if (idx < n) {
    z[idx] = a * x[idx] + y[idx];
  }
}

template <typename T>
void axpy_async(cudaStream_t stream, size_t n, T *z, const T a, const T *x,
                T *y) {
  size_t threads_per_block = 256;
  size_t num_blocks = (n + threads_per_block - 1) / threads_per_block;
  axpy_kernel<T><<<num_blocks, threads_per_block, 0, stream>>>(n, z, a, x, y);
}

template <typename T>
__global__ void damping_kernel(size_t n, T *z, const T a, const T *diag,
                               const T *x) {
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
      static_cast<size_t>(threadIdx.x);
  if (idx < n) {
    z[idx] += a * diag[idx] * x[idx];
  }
}

template <typename T>
void damp_by_factor_async(cudaStream_t stream, size_t n, T *z, const T a,
                          const T *diag, const T *x) {
  size_t threads_per_block = 256;
  size_t num_blocks = (n + threads_per_block - 1) / threads_per_block;
  damping_kernel<T>
      <<<num_blocks, threads_per_block, 0, stream>>>(n, z, a, diag, x);
}

template <typename T>
__global__ void clamp_kernel(size_t n, T min_val, T max_val, T *x) {
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
      static_cast<size_t>(threadIdx.x);
  if (idx < n) {
    x[idx] = std::clamp(x[idx], min_val, max_val);
  }
}

template <typename T>
void clamp_async(cudaStream_t stream, size_t n, T min_val, T max_val, T *x) {
  size_t threads_per_block = 256;
  size_t num_blocks = (n + threads_per_block - 1) / threads_per_block;
  clamp_kernel<T>
      <<<num_blocks, threads_per_block, 0, stream>>>(n, min_val, max_val, x);
}

template <typename T>
__global__ void rescale_vec_kernel(size_t n, T *out, const T scale,
                                   const T *x) {
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
      static_cast<size_t>(threadIdx.x);
  if (idx < n) {
    out[idx] = scale * x[idx];
  }
}

template <typename T>
void rescale_vec_async(cudaStream_t stream, size_t n, T *out, const T scale,
                       const T *x) {
  size_t threads_per_block = 256;
  size_t num_blocks = (n + threads_per_block - 1) / threads_per_block;
  rescale_vec_kernel<T>
      <<<num_blocks, threads_per_block, 0, stream>>>(n, out, scale, x);
}

} // namespace graphite