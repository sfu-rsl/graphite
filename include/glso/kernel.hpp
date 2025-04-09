#pragma once
// #include <cublas_v2.h>

template <typename T>
__global__ 
void saxpy_kernel(size_t n, T* z, const T a, const T* x, T* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = a * x[idx] + y[idx];
    }

}


template <typename T>
void saxpy(size_t n, T* z, const T a, const T* x, T* y) {
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    saxpy_kernel<T><<<num_blocks, threads_per_block>>>(n, z, a, x, y);
    cudaDeviceSynchronize();
}