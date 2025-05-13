#pragma once
#include <glso/vertex.hpp>
#include <glso/factor.hpp>
#include <Eigen/Dense>

namespace glso {

template<typename T>
class Preconditioner {

    virtual void precompute(
        std::vector<BaseVertexDescriptor<T>*>& vertex_descriptors,
            std::vector<BaseFactorDescriptor<T>*>& factor_descriptors, size_t dimension) = 0;

    virtual void apply(T* z, const T* r) = 0;
};

template<typename T>
class IdentityPreconditioner: public Preconditioner<T> {
    private:
        size_t dimension;
    public:
        void precompute(
            std::vector<BaseVertexDescriptor<T>*>& vertex_descriptors,
            std::vector<BaseFactorDescriptor<T>*>& factor_descriptors, size_t dimension) override {
                this->dimension = dimension;
            }

        void apply(T* z, const T* r) override {
            // thrust::copy(r, r + dimension, z);
            cudaMemcpy(z, r, dimension*sizeof(T), cudaMemcpyDeviceToDevice);
        }

};

template<typename T, int I, int N, int E, int D>
__global__ void compute_hessian_diagonal_kernel(
    T* diagonal_blocks, const T* jacs, const size_t* ids, const uint32_t* fixed, 
    const T* pmat, const T* chi2_derivative, const size_t num_threads) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_threads) {
            return;
        }

    constexpr auto jacobian_size = D*E;
    constexpr auto block_size = D*D;

    // Stored as E x d col major, but we need to transpose it to d x E, where d is the vertex size
    const size_t factor_id = idx / block_size;
    const size_t local_id = ids[factor_id*N+I]; // N is the number of vertices involved in the factor
    if (is_fixed(fixed, local_id)) {
        return;
    }
    const auto jacobian_offset = factor_id * jacobian_size;

    constexpr auto precision_matrix_size = E*E;
    const auto precision_offset = factor_id*precision_matrix_size;
    
    // Identify H block row and column (column major)
    const size_t row = idx % D;
    const size_t col = idx / D;

    // left[i]*pmat[i*E+j]*right[i] = h value
    // where i goes from 0 to E
    const T* Jt = jacs + jacobian_offset + row*E;
    const T* J = jacs + jacobian_offset + col*E;

    const T* precision_matrix = pmat + precision_offset;

    T value = 0;
    #pragma unroll
    for (int i = 0; i < E; i++) { // pmat row
        #pragma unroll
        for (int j = 0; j < E; j++) { // pmat col
            value += Jt[i]*J[i]*precision_matrix[i*E + j];
        }
    }
    value *= chi2_derivative[factor_id];


    T* block = diagonal_blocks + local_id*block_size + (idx % block_size);
    
    atomicAdd(block, value);

}

template<typename T, int I, int N, int E, int D>
__global__ void invert_hessian_diagonal_kernel(
    T* diagonal_blocks, const uint32_t* fixed, const T* chi2_derivative, const size_t num_threads) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_threads) {
            return;
        }


    constexpr auto block_size = D*D;

    const auto vertex_id = idx;
    if (is_fixed(fixed, vertex_id)) {
        return;
    }

    T* block = diagonal_blocks + vertex_id*block_size;

    Eigen::Map<Eigen::Matrix<T, D, D>> block_matrix(block);
    block_matrix = block_matrix.inverse();
}


template<typename T>
class BlockJacobi: public Preconditioner<T> {
    private:
        size_t dimension;
        std::vector<std::pair<size_t, size_t>> block_sizes;
        std::unordered_map<BaseVertexDescriptor<T>*, thrust::device_vector<T>> block_diagonals;
    public:
        void precompute(
            std::vector<BaseVertexDescriptor<T>*>& vertex_descriptors,
            std::vector<BaseFactorDescriptor<T>*>& factor_descriptors, size_t dimension) override {
                this->dimension = dimension;

                for (auto & desc: vertex_descriptors) {
                    // Reserve space
                    const auto d = desc->dimension();
                    const size_t num_values = d*d*desc->count(); // this is not tightly packed since count includes fixed vertices
                    block_diagonals[desc] = thrust::device_vector<T>(num_values, 0);
                }

                // Compute Hessian blocks on the diagonal
                for (auto & desc: factor_descriptors) {
                }

                // Invert the blocks
                for (auto & desc: vertex_descriptors) {
                }

                cudaDeviceSynchronize();

            }
};

}