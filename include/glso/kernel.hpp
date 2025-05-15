#pragma once
// #include <cublas_v2.h>
namespace glso {
template <typename T>
__global__ 
void axpy_kernel(size_t n, T* z, const T a, const T* x, T* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = a * x[idx] + y[idx];
    }

}


template <typename T>
void axpy(size_t n, T* z, const T a, const T* x, T* y) {
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    axpy_kernel<T><<<num_blocks, threads_per_block>>>(n, z, a, x, y);
    cudaDeviceSynchronize();
}

template <typename T>
__global__ 
void damping_kernel(size_t n, T* z, const T a, const T* diag, const T* x, T* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = a * diag[idx] * x[idx] + y[idx];
    }
}

template <typename T>
void damp_by_factor(size_t n, T* z, const T a, const T* diag, const T* x, T* y) {
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    damping_kernel<T><<<num_blocks, threads_per_block>>>(n, z, a, diag, x, y);
    cudaDeviceSynchronize();
}

}