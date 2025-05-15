#pragma once
#include <glso/vertex.hpp>
#include <glso/factor.hpp>
#include <Eigen/Dense>
#include <glso/op.hpp>
#include <cublas_v2.h>

namespace glso {

template<typename T>
class Preconditioner {

    virtual void precompute(
        GraphVisitor<T>& visitor,
        std::vector<BaseVertexDescriptor<T>*>& vertex_descriptors,
            std::vector<BaseFactorDescriptor<T>*>& factor_descriptors, size_t dimension, T mu) = 0;

    virtual void apply(GraphVisitor<T>& visitor, T* z, const T* r) = 0;
};

template<typename T>
class IdentityPreconditioner: public Preconditioner<T> {
    private:
        size_t dimension;
    public:
        void precompute(
            GraphVisitor<T>& visitor,
            std::vector<BaseVertexDescriptor<T>*>& vertex_descriptors,
            std::vector<BaseFactorDescriptor<T>*>& factor_descriptors, size_t dimension, T mu) override {
                this->dimension = dimension;
            }

        void apply(GraphVisitor<T>& visitor, T* z, const T* r) override {
            // thrust::copy(r, r + dimension, z);
            cudaMemcpy(z, r, dimension*sizeof(T), cudaMemcpyDeviceToDevice);
        }

};

// template<typename T, int I, int N, int E, int D>
// __global__ void compute_hessian_diagonal_kernel(
//     T* diagonal_blocks, const T* jacs, const size_t* ids, const uint32_t* fixed, 
//     const T* pmat, const T* chi2_derivative, const size_t num_threads) {
//         int idx = blockIdx.x * blockDim.x + threadIdx.x;

//         if (idx >= num_threads) {
//             return;
//         }

//     constexpr auto jacobian_size = D*E;
//     constexpr auto block_size = D*D;

//     // Stored as E x d col major, but we need to transpose it to d x E, where d is the vertex size
//     const size_t factor_id = idx / block_size;
//     const size_t local_id = ids[factor_id*N+I]; // N is the number of vertices involved in the factor
//     if (is_fixed(fixed, local_id)) {
//         return;
//     }
//     const auto jacobian_offset = factor_id * jacobian_size;

//     constexpr auto precision_matrix_size = E*E;
//     const auto precision_offset = factor_id*precision_matrix_size;
    
//     // Identify H block row and column (column major)
//     const size_t row = idx % D;
//     const size_t col = idx / D;

//     // left[i]*pmat[i*E+j]*right[i] = h value
//     // where i goes from 0 to E
//     const T* Jt = jacs + jacobian_offset + row*E;
//     const T* J = jacs + jacobian_offset + col*E;

//     const T* precision_matrix = pmat + precision_offset;

//     T value = 0;
//     #pragma unroll
//     for (int i = 0; i < E; i++) { // pmat row
//         #pragma unroll
//         for (int j = 0; j < E; j++) { // pmat col
//             value += Jt[i]*J[i]*precision_matrix[i*E + j];
//         }
//     }
//     value *= chi2_derivative[factor_id];


//     T* block = diagonal_blocks + local_id*block_size + (idx % block_size);
    
//     atomicAdd(block, value);

// }

// template<typename T>
// class ComputeHessianDiagonal: public OpImpl<T, ComputeHessianDiagonal<T>> {
//     private:
//     public:

//         template<typename F>
//         void apply_op(Op<T>& op) override {

//             std::cout << "ComputeHessianDiagonal applied!" << std::endl;
//             std::cout << "Operating on " << typeid(F).name() << "\n";

//             // auto* compute_hessian = dynamic_cast<ComputeHessianDiagonal<T>*>(&op);
//             // if (compute_hessian) {
//             //     compute_hessian->apply_op(op);
//             // }
//         }
// };

// template<typename T, int I, int N, int E, int D>
// __global__ void invert_hessian_diagonal_kernel(
//     T* diagonal_blocks, const uint32_t* fixed, const T* chi2_derivative, const size_t num_threads) {
//         int idx = blockIdx.x * blockDim.x + threadIdx.x;

//         if (idx >= num_threads) {
//             return;
//         }


//     constexpr auto block_size = D*D;

//     const auto vertex_id = idx;
//     if (is_fixed(fixed, vertex_id)) {
//         return;
//     }

//     T* block = diagonal_blocks + vertex_id*block_size;

//     Eigen::Map<Eigen::Matrix<T, D, D>> block_matrix(block);
//     block_matrix = block_matrix.inverse();
// }


template<typename T>
class BlockJacobiPreconditioner: public Preconditioner<T> {
    private:
        size_t dimension;
        std::vector<std::pair<size_t, size_t>> block_sizes;
        std::unordered_map<BaseVertexDescriptor<T>*, thrust::device_vector<T>> block_diagonals;
        std::vector<BaseVertexDescriptor<T>*>* vds;
        cublasHandle_t handle;

    public:

        BlockJacobiPreconditioner() {
            cublasCreate(&handle);
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        }

        void precompute(
            GraphVisitor<T>& visitor,
            std::vector<BaseVertexDescriptor<T>*>& vertex_descriptors,
            std::vector<BaseFactorDescriptor<T>*>& factor_descriptors, size_t dimension,
            T mu) override {
                this->dimension = dimension;
                this->vds = &vertex_descriptors;

                for (auto & desc: vertex_descriptors) {
                    // Reserve space
                    const auto d = desc->dimension();
                    const size_t num_values = d*d*desc->count(); // this is not tightly packed since count includes fixed vertices
                    block_diagonals[desc] = thrust::device_vector<T>(num_values, 0);
                    // block_diagonals.insert(desc, thrust::device_vector<T>(num_values, 0));
                }

                // Compute Hessian blocks on the diagonal
                for (auto & desc: vertex_descriptors) {
                    thrust::fill(block_diagonals[desc].begin(), block_diagonals[desc].end(), 0);
                }
                for (auto & desc: factor_descriptors) {
                    desc->visit_block_diagonal(visitor, block_diagonals);
                }
                cudaDeviceSynchronize();

                // Print first 5 blocks for each vertex desc
                // for (auto & desc: vertex_descriptors) {
                //     const auto d = desc->dimension();
                //     // T* blocks = block_diagonals[desc].data().get();
                //     thrust::host_vector<T> blocks = block_diagonals[desc];
                //     const size_t num_values = desc->count(); // this is not tightly packed since count includes fixed vertices
                //     T* p_blocks = blocks.data();
                //     std::cout << "Block diagonal for vertex descriptor " << desc << ": ";
                //     for (size_t i = 0; i < std::min(num_values, size_t(1)); i++) {
                //         // std::cout << blocks[i] << " ";
                //         auto map = Eigen::Map<Eigen::MatrixXd>(p_blocks + i*d*d, d, d);
                //         std::cout << "Matrix:\n" << map << std::endl;
                //     }
                //     std::cout << std::endl;
                // }

                // Invert the blocks
                 // TODO: Figure out a better way to handle the memory
                thrust::device_vector<T> Ainv_data;
                thrust::host_vector<T*> A_ptrs, Ainv_ptrs;
                thrust::device_vector<T*> A_ptrs_device, Ainv_ptrs_device;
                thrust::device_vector<int> info;
                for (auto & desc: vertex_descriptors) {
                    desc->visit_augment_block_diagonal(visitor, block_diagonals[desc].data().get(), mu);
                    // Invert the block diagonal using cublas
                    const auto d = desc->dimension();
                    const size_t num_blocks = desc->count();
                    const auto block_size = d*d;

                   
                    A_ptrs.resize(num_blocks);
                    Ainv_ptrs.resize(num_blocks);
                    Ainv_data.resize(num_blocks * block_size);
                    info.resize(num_blocks);

                    T* a_ptr = block_diagonals[desc].data().get();
                    T* a_inv_ptr = Ainv_data.data().get();
                    for (size_t i = 0; i < num_blocks; ++i) {
                        A_ptrs[i] = a_ptr + i * block_size;
                        Ainv_ptrs[i] = a_inv_ptr + i * block_size;
                    }

                    A_ptrs_device = A_ptrs;
                    Ainv_ptrs_device = Ainv_ptrs;


                    cublasDmatinvBatched(handle, d, A_ptrs_device.data().get(), 
                    d, Ainv_ptrs_device.data().get(), d, info.data().get(), num_blocks);

                    cudaDeviceSynchronize();

                    // Copy back
                    block_diagonals[desc] = Ainv_data;
                }


                

                // for (auto & desc: vertex_descriptors) {
                //     const auto d = desc->dimension();
                //     const size_t num_blocks = desc->count();
                //     thrust::host_vector<T> blocks_host = block_diagonals[desc];
                //     T* p_blocks = blocks_host.data();
                //     for (size_t i = 0; i < num_blocks; ++i) {
                //         Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat(p_blocks + i * d * d, d, d);
                //         mat = mat.inverse();
                //     }
                //     block_diagonals[desc] = blocks_host;
                // }

                // for (auto & desc: vertex_descriptors) {
                //     const auto d = desc->dimension();
                //     // T* blocks = block_diagonals[desc].data().get();
                //     thrust::host_vector<T> blocks = block_diagonals[desc];
                //     const size_t num_values = desc->count(); // this is not tightly packed since count includes fixed vertices
                //     T* p_blocks = blocks.data();
                //     std::cout << "(post inversion) Block diagonal for vertex descriptor " << desc << ": ";
                //     for (size_t i = 0; i < std::min(num_values, size_t(2)); i++) {
                //         // std::cout << blocks[i] << " ";
                //         auto map = Eigen::Map<Eigen::MatrixXd>(p_blocks + i*d*d, d, d);
                //         std::cout << "Matrix:\n" << map << std::endl;
                //                                 Eigen::MatrixXd inv = map.inverse();

                //         // std::cout << "Inverse:\n" << inv << std::endl;
                        
                //     }
                //     std::cout << std::endl;
                // }

                cudaDeviceSynchronize();

            }

            void apply(GraphVisitor<T>& visitor, T* z, const T* r) override {
                // Apply the preconditioner
                for (auto & desc: *vds) {
                    const auto d = desc->dimension();
                    T* blocks = block_diagonals[desc].data().get();
                    // std::cout << "bd size: " << block_diagonals[desc].size() << std::endl;
                    desc->visit_apply_block_jacobi(visitor, z, r, blocks);
                }
                cudaDeviceSynchronize();
            }
};

}