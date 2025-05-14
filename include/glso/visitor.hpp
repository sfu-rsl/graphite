#pragma once
#include <glso/common.hpp>
#include <Eigen/Dense>

namespace glso {

    template <typename VPtr, typename T, size_t D>
    __device__ void device_copy(const VPtr v, T* dst) {
        const std::array<T, D> src = v->parameters();
        #pragma unroll
        for (size_t i = 0; i < D; i++) {
            dst[i] = src[i];
        }
    }

    template <typename VPtr, typename T, size_t D>
    __device__ void real_to_dual(const VPtr v, Dual<T>* dst) {
        const std::array<T, D> src = v->parameters();
        #pragma unroll
        for (size_t i = 0; i < D; i++) {
            dst[i] = Dual<T>(src[i]);
        }
    }

    __device__ bool is_fixed(const uint32_t* fixed, const size_t vertex_id) {
        const uint32_t mask = 1 << (vertex_id % 32);
        return (fixed[vertex_id / 32] & mask);
    }

    template<typename T, size_t E>
    __device__ T compute_chi2(const T* residuals, const T* pmat, const size_t factor_id) {
        T r2[E] = {0};
    
        #pragma unroll
        for (int i = 0; i < E; i++) {
            #pragma unroll
            for (int j = 0; j < E; j++) {
                r2[i] += pmat[factor_id*E*E + i*E + j] * residuals[factor_id*E + j];
            }
        }
    
        T value = 0;
        #pragma unroll
        for (int i = 0; i < E; i++) {
            value += r2[i] * residuals[factor_id*E + i];
        }
    
        return value;
    }

template<typename T, typename Descriptor, typename V>
__global__ void apply_update_kernel(V** vertices, const T* delta_x, const size_t * hessian_ids, const uint32_t* fixed, const size_t num_threads) {
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex_id >= num_threads || is_fixed(fixed, vertex_id)) {
        return;
    }

    const T* delta = delta_x + hessian_ids[vertex_id];

    // Descriptor::update(vertices[vertex_id], delta);
    vertices[vertex_id]->update(delta);

}

template<typename T, int D>
__global__ void invert_hessian_diagonal_kernel(
    T* diagonal_blocks, const T mu, const uint32_t* fixed, const size_t num_threads) {
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
        // Eigen::Matrix<T, D, D> mat = block_matrix;
        // #pragma unroll
        // for (int i = 0; i < D; i++) {
        //     #pragma unroll
        //     for (int j = 0; j < D; j++) {
        //         mat(i, j) = block[i + j*D];
        //     }
        // } 
        Eigen::Matrix<T,D, D> mat = (block_matrix+mu*Eigen::Matrix<T, D, D>::Identity()).eval();
        block_matrix = mat.eval();
        // mat = (mat+mu*Eigen::Matrix<T, D, D>::Identity()).eval();
        // mat = Eigen::Matrix<T, D, D>::Identity();
        // Eigen::Matrix<T, D, D> mat = (Eigen::Matrix<T, D, D>::Identity()).eval(); 
        // Eigen::Matrix<T, D, D> mat_inv = mat.inverse().eval();
        // block_matrix = mat_inv;
        
        // block_matrix = mat;
        // block_matrix = Eigen::Matrix<T, D, D>::Identity().eval();

        // #pragma unroll
        // for (int i = 0; i < D; i++) {
        //     #pragma unroll
        //     for (int j = 0; j < D; j++) {
        //         block[i + j*D] = mat(i, j);
        //     }
        // }
}

template<typename T, int D>
__global__ void apply_block_jacobi_kernel(
    T* z, const T* r, T* block_diagonal, const size_t* hessian_ids, const uint32_t* fixed, const size_t num_threads) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        const auto local_vertex_id = idx / D;

        if (idx >= num_threads || is_fixed(fixed, local_vertex_id)) {
            return;
        }

        constexpr auto block_size = D*D;

        
        T* block = block_diagonal + local_vertex_id*block_size;
        const auto hessian_offset = hessian_ids[local_vertex_id];
        const auto offset = idx % D;
        const auto row = offset;

        Eigen::Map<Eigen::Matrix<T, D, D>> block_matrix(block);
        // Eigen::Map<Eigen::Matrix<T, D, 1>> z_map(z + hessian_offset + offset);
        // Eigen::Map<const Eigen::Matrix<T, D, 1>> r_map(r + hessian_offset + offset);   
        // z_map += block_matrix*r_map;
        // block_matrix = Eigen::Matrix<T, D, D>::Identity();
        T value = 0;
        // #pragma unroll
        // for (size_t i = 0; i < D; i++) {
        //     value += block[row + i*D] * r[hessian_offset + i];
        // }
        // value  = r[hessian_offset + row];
        // value  = (1.0/block[row+row*D])*r[hessian_offset + row];
        value = (1.0/block_matrix(row, row))*r[hessian_offset + row];
        // atomicAdd(z + hessian_offset + row, value);
        z[hessian_offset + row] = value;
}

template<typename T, size_t I, size_t N, typename M, size_t E, typename F, typename VT, std::size_t... Is>
__global__
void compute_error_kernel_autodiff(const M* obs, T* error, const typename F::ConstraintDataType* constraint_data, size_t* ids, const size_t* hessian_ids, const size_t num_threads, VT args, std::array<T*, sizeof...(Is)> jacs, const uint32_t* fixed, std::index_sequence<Is...>) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_threads) {
        return;
    }


    constexpr auto vertex_sizes = F::get_vertex_sizes();
    const auto factor_id = idx / vertex_sizes[I];
    const auto vertex_id = ids[factor_id*N + I];

    // printf("CEAD: Thread %d, Vertex %d, Factor %d\n", idx, vertex_id, factor_id);
    
    const M* local_obs = obs + factor_id;
    Dual<T> local_error[E];
    const typename F::ConstraintDataType* local_data = constraint_data + factor_id;

    #pragma unroll
    for (int i = 0; i < E; ++i) {
        local_error[i] = error[factor_id * E + i];
    }


    auto v = cuda::std::make_tuple(std::array<Dual<T>, vertex_sizes[Is]>{}...);
    

    auto vargs = std::make_tuple((*(std::get<Is>(args) + ids[factor_id*N+Is]))...);

    auto copy_vertices = [&v, &vertex_sizes, &vargs](auto&&... ptrs) {
        ((real_to_dual<decltype(std::get<Is>(vargs)), T, vertex_sizes[Is]>(std::get<Is>(vargs), cuda::std::get<Is>(v).data())), ...);
    };

    std::apply(copy_vertices, vargs);

    cuda::std::get<I>(v)[idx % vertex_sizes[I]].dual = static_cast<T>(1);



    F::error(cuda::std::get<Is>(v).data()..., local_obs, local_error, vargs, local_data);


    constexpr auto j_size = vertex_sizes[I]*E;
    // constexpr auto col_offset = I*E;
    const auto col_offset = (idx % vertex_sizes[I])*E;
    // Store column-major Jacobian blocks.
    // Write one scalar column (length E) of the Jacobian matrix.
    // TODO: make sure this only writes to each location once
    // The Jacobian is stored as E x vertex_size in col major

    // Only run once per factor - this check won't work for multiple kernel launches
    // TODO: make sure this only writes to each location once for the error
    if (idx % vertex_sizes[I] == 0) {
        #pragma unroll
        for(size_t i = 0; i < E; ++i) {
            error[factor_id * E + i] = local_error[i].real;
            // printf("Error[%d] = %f\n", factor_id * E + i, error[factor_id * E + i]);
        }
    }

    // This should write one Jacobian column per dimension per vertex for each factor
    // We only need a Jacobian if the vertex is not fixed
    if (is_fixed(fixed, vertex_id)) {
        return;
    }
    #pragma unroll
    for(size_t i = 0; i < E; ++i) {
        jacs[I][j_size*factor_id + col_offset + i] = local_error[i].dual;
        // printf("Jacobian[%d] = %f\n", j_size*factor_id + col_offset + i, jacs[I][j_size*factor_id + col_offset + i]);
    }
}
// TODO: Make this more efficient and see if code can be shared with the autodiff kernel
template<typename T, size_t N, typename M, size_t E, typename F, typename VT, std::size_t... Is>
__global__
void compute_error_kernel(const M* obs, T* error, const typename F::ConstraintDataType* constraint_data, size_t* ids, const size_t num_threads, VT args, std::index_sequence<Is...>) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_threads) {
        return;
    }

    constexpr auto vertex_sizes = F::get_vertex_sizes();
    const auto factor_id = idx;


    const M* local_obs = obs + factor_id;
    T* local_error = error + factor_id * E;
    const typename F::ConstraintDataType* local_data = constraint_data + factor_id;

    auto v = cuda::std::make_tuple(std::array<T, vertex_sizes[Is]>{}...);

    auto vargs = std::make_tuple((*(std::get<Is>(args) + ids[factor_id*N+Is]))...);

    auto copy_vertices = [&v, &ids, &vertex_sizes, &vargs](auto&&... ptrs) {
        ((device_copy<decltype(std::get<Is>(vargs)), T, vertex_sizes[Is]>(std::get<Is>(vargs), cuda::std::get<Is>(v).data())), ...);
    };

    std::apply(copy_vertices, vargs);

    F::error(cuda::std::get<Is>(v).data()..., local_obs, local_error, vargs, local_data);


    #pragma unroll
    for(size_t i = 0; i < E; ++i) {
        error[factor_id * E + i] = local_error[i];
    }

}

// The output will be part of b with length of the vertex (where b = -J^T * r)
// Note the negative sign - different papers use different conventions
// TODO: Replace with generic J^T x r kernel?
// Note: The error vector is local to the factor
template<typename T, size_t I, size_t N, size_t E, typename F, std::size_t... Is>
__global__ 
void compute_b_kernel_no_precision_matrix(T* b, T* error, size_t* ids, const size_t* hessian_ids, const size_t num_threads, std::array<T*, sizeof...(Is)> jacs, std::index_sequence<Is...>) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_threads) {
        return;
    }

    constexpr auto vertex_sizes = F::get_vertex_sizes();
    constexpr auto jacobian_size = vertex_sizes[I]*E;
    
    // Stored as E x d col major, but we need to transpose it to d x E, where d is the vertex size
    const size_t factor_id = idx / vertex_sizes[I];
    const auto jacobian_offset = factor_id * jacobian_size;
    const auto error_offset = factor_id*E;
    // constexpr auto col_offset = I*E; // for untransposed J
    const auto col_offset = (idx % vertex_sizes[I])*E; // for untransposed J


    T value = 0;

    #pragma unroll
    for (int i = 0; i < E; i++) {
        value -= jacs[I][jacobian_offset + col_offset + i] * error[error_offset + i];
    }

    size_t local_id = ids[factor_id*N+I]; // N is the number of vertices involved in the factor
    const auto hessian_offset = hessian_ids[local_id]; // each vertex has a hessian_ids array
    
    // printf("Hessian offset: %u\n", hessian_offset);
    // printf("Adding b[%d] += %f\n", hessian_offset + (idx % vertex_sizes[I]), value);
    // printf("Thread %d, Hessian offset: %u\n", idx, hessian_offset);

    atomicAdd(&b[hessian_offset + (idx % vertex_sizes[I])], value);

}

// Include precision matrix
template<typename T, size_t I, size_t N, size_t E, typename F, typename L, std::size_t... Is>
__global__ 
void compute_b_kernel(T* b, T* error, size_t* ids, const size_t* hessian_ids, const size_t num_threads, T* jacs, const uint32_t* fixed, const T* pmat, const L* loss, std::index_sequence<Is...>) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_threads) {
        return;
    }

    constexpr auto vertex_sizes = F::get_vertex_sizes();
    constexpr auto jacobian_size = vertex_sizes[I]*E;
    
    // Stored as E x d col major, but we need to transpose it to d x E, where d is the vertex size
    const size_t factor_id = idx / vertex_sizes[I];
    const size_t local_id = ids[factor_id*N+I]; // N is the number of vertices involved in the factor
    if (is_fixed(fixed, local_id)) {
        return;
    }
    const auto jacobian_offset = factor_id * jacobian_size;
    const auto error_offset = factor_id*E;
    // constexpr auto col_offset = I*E; // for untransposed J
    const auto col_offset = (idx % vertex_sizes[I])*E; // for untransposed J

    // Use loss kernel
    const auto chi2 = compute_chi2<T, E>(error, pmat, factor_id);
    const auto dL = loss[factor_id].loss_derivative(chi2);

    T value = 0;
    constexpr auto precision_matrix_size = E*E;
    const auto precision_offset = factor_id*precision_matrix_size;
    T x2[E] = {0};

    #pragma unroll
    for (int i = 0; i < E; i++) { // pmat row
        #pragma unroll
        for (int j = 0; j < E; j++) { // pmat col
            // x2[i] += pmat[precision_offset + i + j*E] * error[error_offset + j]; // col major
            x2[i] += dL*pmat[precision_offset + i*E + j] * error[error_offset + j]; // row major (use for faster access on symmetrical matrix)
        }
    }

    #pragma unroll
    for (int i = 0; i < E; i++) {
        value -= jacs[jacobian_offset + col_offset + i] * x2[i];
    }

    const auto hessian_offset = hessian_ids[local_id]; // each vertex has a hessian_ids array
    
    // printf("Hessian offset: %u\n", hessian_offset);
    // printf("Adding b[%d] += %f\n", hessian_offset + (idx % vertex_sizes[I]), value);
    // printf("Thread %d, Hessian offset: %u\n", idx, hessian_offset);

    atomicAdd(&b[hessian_offset + (idx % vertex_sizes[I])], value);

}

// Compute J * x where the length of vector x matches the Hessian dimension
// Each Jacobian block needs to be accessed just once
// So we need E threads for each block (error dimension)
// In total we should hae E*num_factors threads?
template<typename T, size_t I, size_t N, size_t E, size_t D, typename F, std::size_t... Is>
__global__ 
void compute_Jv_kernel(T*y, T* x, T* error, size_t* ids, const size_t* hessian_ids, const size_t num_threads, const T* jacs, const uint32_t* fixed, std::index_sequence<Is...>) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_threads) {
        return;
    }

    constexpr auto jacobian_size = D*E;
    
    // Each J block is stored as E x d col major, where d is the vertex size
    const size_t factor_id = idx / E;
    const size_t local_id = ids[factor_id*N+I]; // N is the number of vertices involved in the factor
    if (is_fixed(fixed, local_id)) {
        return;
    }
    const auto jacobian_offset = factor_id * jacobian_size;

    T value = 0;

    const auto hessian_offset = hessian_ids[local_id]; // each vertex has a hessian_ids array
    const auto row_offset = (idx % E);
    // Adding i*E skips to the next column
    // size_t residual_offset = 0; // need to pass this in
    // it's the offset into the r vector
    // #pragma unroll
    // for (int i = 0; i < d; i++) {
    //     value += jacs[jacobian_offset + row_offset + i*E] * x[hessian_offset + i];
    // }

    const T* jrow = jacs + jacobian_offset + row_offset;
    const T* x_start = x + hessian_offset;

    #pragma unroll
    for (int i = 0; i < D; i++) {
        value += jrow[i*E] * x_start[i];
    }
    
    // atomicAdd(&y[residual_offset + idx], value);
    y[idx] += value;

}

// Compute J^T * x where x is the size of the residual vector
// Each Jacobian block needs to be accessed just once
// For each block, we need d threads where d is the vertex size
// We need to load the x vector location for the corresponding block row of J
// So this assumes that the x vector has the same layout as the residual vector for this factor (rather than a global residual vector)
// The aggregate output will be H x len(x) where H is hessian dimension
template<typename T, size_t I, size_t N, size_t E, typename F, std::size_t... Is>
__global__ 
void compute_Jtv_kernel(T* y, T* x, size_t* ids, const size_t* hessian_ids, const size_t num_threads, T* jacs, std::index_sequence<Is...>) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_threads) {
        return;
    }

    constexpr auto vertex_sizes = F::get_vertex_sizes();
    constexpr auto jacobian_size = vertex_sizes[I]*E;
    
    // Stored as E x d col major, but we need to transpose it to d x E, where d is the vertex size
    const size_t factor_id = idx / vertex_sizes[I];
    const auto jacobian_offset = factor_id * jacobian_size;
    const auto error_offset = factor_id*E;
    const auto col_offset = (idx % vertex_sizes[I])*E; // for untransposed J


    T value = 0;

    #pragma unroll
    for (int i = 0; i < E; i++) {
        value += jacs[jacobian_offset + col_offset + i] * x[error_offset + i];
    }

    size_t local_id = ids[factor_id*N+I]; // N is the number of vertices involved in the factor
    const auto hessian_offset = hessian_ids[local_id]; // each vertex has a hessian_ids array
    
    atomicAdd(&y[hessian_offset + (idx % vertex_sizes[I])], value);

}

// Compute J^T * P * x where P is the precision matrix
template<typename T, size_t I, size_t N, size_t E, size_t D, typename F, std::size_t... Is>
__global__ 
void compute_JtPv_kernel(T* y, const T* x, const size_t* ids, const size_t* hessian_ids, const size_t num_threads, const T* jacs, const uint32_t* fixed, const T* pmat, const T* chi2_derivative, std::index_sequence<Is...>) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_threads) {
        return;
    }

    constexpr auto jacobian_size = D*E;
    
    // Stored as E x d col major, but we need to transpose it to d x E, where d is the vertex size
    const size_t factor_id = idx / D;
    const size_t local_id = ids[factor_id*N+I]; // N is the number of vertices involved in the factor
    if (is_fixed(fixed, local_id)) {
        return;
    }
    const auto jacobian_offset = factor_id * jacobian_size;
    const auto error_offset = factor_id*E;
    const auto col_offset = (idx % D)*E; // for untransposed J

    constexpr auto precision_matrix_size = E*E;
    const auto precision_offset = factor_id*precision_matrix_size;

    // Use loss kernel
    // const auto dL = chi2_derivative[factor_id];

    // T x2[E] = {0};
    // T value = 0;

    const T* jcol = jacs + jacobian_offset + col_offset;

    const T* precision_matrix = pmat + precision_offset;
    const T* x_start = x + error_offset;

    // precision matrices are column major (but should be symmetrical?)
    // #pragma unroll
    // for (int i = 0; i < E; i++) { // pmat row
    //     #pragma unroll
    //     for (int j = 0; j < E; j++) { // pmat col
    //         // x2[i] += pmat[precision_offset + i + j*E] * x[error_offset + j]; // col major
    //         // x2[i] += pmat[precision_offset + i*E + j] * x[error_offset + j]; // row major
    //         x2[i] += precision_matrix[i*E + j] * x_start[j]; // row major
    //     }
    //     // x2[i] *= dL;
    //     // value += dL*jacs[jacobian_offset + col_offset + i] * x2[i];
    //     value += jcol[i] * x2[i];
    // }
    // value *= dL;

    T value = 0;
    #pragma unroll
    for (int i = 0; i < E; i++) { // pmat row
        #pragma unroll
        for (int j = 0; j < E; j++) { // pmat col
            // x2[i] += pmat[precision_offset + i + j*E] * x[error_offset + j]; // col major
            // x2[i] += pmat[precision_offset + i*E + j] * x[error_offset + j]; // row major
            // x2[i] += precision_matrix[i*E + j] * x_start[j]; // row major
            value += jcol[i]* precision_matrix[i*E + j]*x_start[j];
        }
        // x2[i] *= dL;
        // value += dL*jacs[jacobian_offset + col_offset + i] * x2[i];
        // value += jcol[i] * x2[i];
        // value += jcol[i] * x[i];
    }
    value *= chi2_derivative[factor_id];


    // T value = 0;
    // #pragma unroll
    // for (int i = 0; i < E; i++) {
    //     value += jacs[jacobian_offset + col_offset + i] * x2[i];
    // }

    const auto hessian_offset = hessian_ids[local_id]; // each vertex has a hessian_ids array
    
    atomicAdd(&y[hessian_offset + (idx % D)], value);

}

template<typename T, size_t E, typename L>
__global__ 
void compute_chi2_kernel(T* chi2, T* chi2_derivative, const T* residuals, const size_t num_threads, const T* pmat, const L* loss) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_threads) {
        return;
    }
    T raw_chi2 = compute_chi2<T, E>(residuals, pmat, idx);
    chi2[idx] = loss[idx].loss(raw_chi2);
    chi2_derivative[idx] = loss[idx].loss_derivative(raw_chi2);

}

template<typename T, size_t I, size_t N, size_t E, size_t D>
__global__ void compute_hessian_diagonal_kernel(
    T* diagonal_blocks, const T* jacs, 
    const size_t* ids, const uint32_t* fixed, 
    const T* pmat, const T* chi2_derivative, const size_t num_threads) {
        size_t idx = ((size_t)blockIdx.x) * ((size_t)blockDim.x) + ((size_t)threadIdx.x);

        if (idx >= num_threads) {
            return;
        }

    constexpr size_t jacobian_size = D*E;
    constexpr size_t block_size = D*D;

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
    // const size_t row = idx % D;
    // const size_t col = idx / D;

    const size_t offset = idx % block_size;
    const size_t row = offset % D;
    const size_t col = offset / D;

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
        // value += Jt[i]*J[i];
    }

    // if (row == col) {
    //     value = 1;
    //     // printf("Thread %d, row: %d, col: %d, value: %f\n", idx, row, col, value);
    //     // printf("D=%d\n", D);
    // }
    // else {
    //     value = 0;
    // }
    value *= chi2_derivative[factor_id];

    // T* block = diagonal_blocks + local_id*block_size + (idx % block_size);
    // printf("Thread %d, Hessian offset: %u\n", idx, local_id);
    // if (value != 0) {
    //  printf("Thread %d, row: %d, col: %d, value: %f\n", idx, row, col, value);
    // }
    // if (D == 9 && local_id*block_size + row + col*D > 1701) {
    //     printf("Thread %d, vertex id: %u, row: %d, col: %d, value: %f, offset: %u \n", idx, local_id, row, col, value, local_id*block_size + row + col*D);

    // }
    // value = 5.0;
    // T* block = diagonal_blocks + local_id*block_size + row + col*D;
    T* block = diagonal_blocks+(local_id*block_size + row + col*D);
    // if (row == col) {
    //     // local_id = 5;
    //     // row = 6; col = 7;
    //     // value = 8.5;
    //     printf("Thread %d, vertex id: %llu, row: %llu, col: %llu, value: %f, offset: %u \n", idx, local_id, row, col, value, local_id*block_size + row + col*D);
    // }
    // if (row == 8 && col == 8) {
    //     printf("Number of threads: %d\n", num_threads);
    //     printf("Thread %d, vertex id: %u, row: %zu, col: %zu, value: %f\n", idx, local_id, row, col, value);
    // }
    // T* block = diagonal_blocks + row*D + col;
    atomicAdd(block, value);
    // block[0] = value;
    // *block = value;

}

template<typename T>
class GraphVisitor {
private:

thrust::device_vector<T>& delta_x;
thrust::device_vector<T>& b;

template <typename F, typename VT, std::size_t... Is>
void launch_kernel_autodiff(F* f, std::array<const size_t*, F::get_num_vertices()>& hessian_ids, VT & verts, std::array<T*, F::get_num_vertices()> & jacs, const size_t num_factors, std::index_sequence<Is...>) {
            (([&] {
            constexpr auto num_vertices = F::get_num_vertices();
            const auto num_threads = num_factors * F::get_vertex_sizes()[Is];
            // std::cout << "Launching autodiff kernel" << std::endl;
            // std::cout << "Num threads: " << num_threads << std::endl;
            int threads_per_block = 256;
            int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

            // std::cout << "Checking obs ptr: " << f->device_obs.data().get() << std::endl;
            // std::cout << "Checking residual ptr: " << f->residuals.data().get() << std::endl;
            // std::cout << "Checking ids ptr: " << f->device_ids.data().get() << std::endl;

            compute_error_kernel_autodiff<T, Is, num_vertices, typename F::ObservationType, F::error_dim, F, typename F::VertexPointerPointerTuple><<<num_blocks, threads_per_block>>>(
                f->device_obs.data().get(),
                f->residuals.data().get(),
                f->data.data().get(),
                f->device_ids.data().get(),
                hessian_ids[Is],
                num_threads,
                verts,
                jacs,
                f->vertex_descriptors[Is]->get_fixed_mask(),
                std::make_index_sequence<num_vertices>{});
            }()), ...);
        }

template <typename F, std::size_t... Is>
void launch_kernel_compute_b(F* f, T* b, std::array<const size_t*, F::get_num_vertices()>& hessian_ids, std::array<T*, F::get_num_vertices()> & jacs, const size_t num_factors, std::index_sequence<Is...>) {
            (([&] {
            constexpr auto num_vertices = F::get_num_vertices();
            const auto num_threads = num_factors * F::get_vertex_sizes()[Is];
            // std::cout << "Launching compute b kernel" << std::endl;
            // std::cout << "Num threads: " << num_threads << std::endl;
            int threads_per_block = 256;
            int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

            // std::cout << "Checking obs ptr: " << f->device_obs.data().get() << std::endl;
            // std::cout << "Checking residual ptr: " << f->residuals.data().get() << std::endl;
            // std::cout << "Checking ids ptr: " << f->device_ids.data().get() << std::endl;

            compute_b_kernel<T, Is, num_vertices, F::error_dim, F><<<num_blocks, threads_per_block>>>(
                b,
                f->residuals.data().get(),
                f->device_ids.data().get(),
                hessian_ids[Is],
                num_threads,
                jacs[Is],
                f->vertex_descriptors[Is]->get_fixed_mask(),
                f->precision_matrices.data().get(),
                f->loss.data().get(),
                std::make_index_sequence<num_vertices>{});
            }()), ...);
        }



        template <typename F, std::size_t... Is>
        void launch_kernel_compute_JtPv(F* f, T* out, T* in, std::array<const size_t*, F::get_num_vertices()>& hessian_ids, std::array<T*, F::get_num_vertices()> & jacs, const size_t num_factors, std::index_sequence<Is...>) {
                    (([&] {
                    constexpr auto num_vertices = F::get_num_vertices();
                    const auto num_threads = num_factors * F::get_vertex_sizes()[Is];
                    // std::cout << "Launching compute Jtv kernel" << std::endl;
                    // std::cout << "Num threads: " << num_threads << std::endl;
                    int threads_per_block = 256;
                    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
        
                    // std::cout << "Checking obs ptr: " << f->device_obs.data().get() << std::endl;
                    // std::cout << "Checking residual ptr: " << f->residuals.data().get() << std::endl;
                    // std::cout << "Checking ids ptr: " << f->device_ids.data().get() << std::endl;
        
                    compute_JtPv_kernel<T, Is, num_vertices, F::error_dim, f->get_vertex_sizes()[Is], F><<<num_blocks, threads_per_block>>>(
                        out,
                        in,
                        f->device_ids.data().get(),
                        hessian_ids[Is],
                        num_threads,
                        jacs[Is],
                        f->vertex_descriptors[Is]->get_fixed_mask(),
                        f->precision_matrices.data().get(),
                        f->chi2_derivative.data().get(),
                        std::make_index_sequence<num_vertices>{});
                    }()), ...);
                }

        template <typename F, std::size_t... Is>
        void launch_kernel_compute_Jv(F* f, T* out, T* in, std::array<const size_t*, F::get_num_vertices()>& hessian_ids, std::array<T*, F::get_num_vertices()> & jacs, const size_t num_factors, std::index_sequence<Is...>) {
                    (([&] {
                    constexpr auto num_vertices = F::get_num_vertices();
                    const auto num_threads = num_factors * F::error_dim;
                    // std::cout << "Launching compute Jv kernel" << std::endl;
                    // std::cout << "Num threads: " << num_threads << std::endl;
                    int threads_per_block = 256;
                    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
        
                    // std::cout << "Checking obs ptr: " << f->device_obs.data().get() << std::endl;
                    // std::cout << "Checking residual ptr: " << f->residuals.data().get() << std::endl;
                    // std::cout << "Checking ids ptr: " << f->device_ids.data().get() << std::endl;
        
                    compute_Jv_kernel<T, Is, num_vertices, F::error_dim, f->get_vertex_sizes()[Is], F><<<num_blocks, threads_per_block>>>(
                        out,
                        in,
                        f->residuals.data().get(),
                        f->device_ids.data().get(),
                        hessian_ids[Is],
                        num_threads,
                        jacs[Is],
                        f->vertex_descriptors[Is]->get_fixed_mask(),
                        std::make_index_sequence<num_vertices>{});
                    }()), ...);
                }


        template <typename F, std::size_t... Is>
        void launch_kernel_block_diagonal(F* f, 
            std::array<T*,F::get_num_vertices()> & diagonal_blocks,  std::array<const size_t*, F::get_num_vertices()>& hessian_ids, std::array<T*, F::get_num_vertices()> & jacs, const size_t num_factors, std::index_sequence<Is...>) {
            (([&] {
            constexpr size_t num_vertices = F::get_num_vertices();
            constexpr size_t dimension = F::get_vertex_sizes()[Is];
            const size_t num_threads = num_factors * dimension*dimension;
            // std::cout << "Launching block diagonal kernel" << std::endl;
            // std::cout << "Num threads: " << num_threads << std::endl;
            // std::cout << "dimension: " << dimension << std::endl;
            // std::cout << "num_factors: " << num_factors << std::endl;
            int threads_per_block = 256;
            int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

            // std::cout << "Checking obs ptr: " << f->device_obs.data().get() << std::endl;
            // std::cout << "Checking residual ptr: " << f->residuals.data().get() << std::endl;
            // std::cout << "Checking ids ptr: " << f->device_ids.data().get() << std::endl;

            compute_hessian_diagonal_kernel<T, Is, num_vertices, F::error_dim, dimension><<<num_blocks, threads_per_block>>>(
                diagonal_blocks[Is],
                jacs[Is],
                f->device_ids.data().get(),
                f->vertex_descriptors[Is]->get_fixed_mask(),
                f->precision_matrices.data().get(),
                f->chi2_derivative.data().get(),
                num_threads);

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                }
                
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    std::cerr << "CUDA error after kernel execution: " << cudaGetErrorString(err) << std::endl;
                }
            }()), ...);
        }


public:
    
    GraphVisitor(thrust::device_vector<T>& delta_x, thrust::device_vector<T>& b): delta_x(delta_x), b(b) {
    }

    template<typename F, typename... VertexTypes>
    void compute_error_autodiff(F* f) {
        // Assume autodiff

        // Then for each vertex, we need to compute the error
        constexpr auto num_vertices = F::get_num_vertices();
        constexpr auto vertex_sizes = F::get_vertex_sizes();

        // At this point all necessary data should be on the GPU    
        // std::array<T*, num_vertices> verts;
        auto verts = f->get_vertices();
        std::array<T*, num_vertices> jacs;
        std::array<const size_t*, num_vertices> hessian_ids;
        for (int i = 0; i < num_vertices; i++) {
            // verts[i] = f->vertex_descriptors[i]->vertices();
            jacs[i] = f->jacobians[i].data.data().get();
            hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();

            // Important: Must clear Jacobian storage
            thrust::fill(f->jacobians[i].data.begin(), f->jacobians[i].data.end(), 0);
        }

        
        const auto num_factors = f->count();

        launch_kernel_autodiff(f, hessian_ids, verts, jacs, num_factors, std::make_index_sequence<num_vertices>{});
        cudaDeviceSynchronize();

    }

    template<typename F, typename... VertexTypes>
    void compute_error(F* f) {
        // Then for each vertex, we need to compute the error
        constexpr auto num_vertices = F::get_num_vertices();
        constexpr auto vertex_sizes = F::get_vertex_sizes();

        // At this point all necessary data should be on the GPU    
        auto verts = f->get_vertices();
        // std::array<T*, num_vertices> verts;
        std::array<const size_t*, num_vertices> hessian_ids;
        for (int i = 0; i < num_vertices; i++) {
            // verts[i] = f->vertex_descriptors[i]->x();
            hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
        }

        
        constexpr auto error_dim = F::error_dim;
        const auto num_factors = f->count();

        const auto num_threads = num_factors;
        int threads_per_block = 256;
        int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

        compute_error_kernel<T, num_vertices, typename F::ObservationType, F::error_dim, F, typename F::VertexPointerPointerTuple><<<num_blocks, threads_per_block>>>(
            f->device_obs.data().get(),
            f->residuals.data().get(),
            f->data.data().get(),
            f->device_ids.data().get(),
            num_threads,
            verts,
            std::make_index_sequence<num_vertices>{});

        cudaDeviceSynchronize();

    }

    template<typename F>
    void compute_chi2(F* f) {
        // Then for each vertex, we need to compute the error
        constexpr auto num_vertices = F::get_num_vertices();
        constexpr auto vertex_sizes = F::get_vertex_sizes();

        // At this point all necessary data should be on the GPU    
        auto verts = f->get_vertices();
        // std::array<T*, num_vertices> verts;
        // std::array<const size_t*, num_vertices> hessian_ids;
        // for (int i = 0; i < num_vertices; i++) {
            // verts[i] = f->vertex_descriptors[i]->x();
            // hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
        // }

        
        constexpr auto error_dim = F::error_dim;
        const auto num_factors = f->count();

        const auto num_threads = num_factors;
        int threads_per_block = 256;
        int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

        compute_chi2_kernel<T, F::error_dim><<<num_blocks, threads_per_block>>>(
            f->chi2_vec.data().get(),
            f->chi2_derivative.data().get(),
            f->residuals.data().get(),
            num_threads,
            f->precision_matrices.data().get(),
            f->loss.data().get()
        );

        cudaDeviceSynchronize();

    }

    template<typename F, typename... VertexTypes>
    void compute_b(F* f) {
        constexpr auto num_vertices = F::get_num_vertices();
        constexpr auto vertex_sizes = F::get_vertex_sizes();

        // std::array<T*, num_vertices> verts;
        auto verts = f->get_vertices();
        std::array<T*, num_vertices> jacs;
        std::array<const size_t*, num_vertices> hessian_ids;
        for (int i = 0; i < num_vertices; i++) {
            // verts[i] = f->vertex_descriptors[i]->x();
            jacs[i] = f->jacobians[i].data.data().get();
            hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
        }

        
        constexpr auto error_dim = F::error_dim;
        const auto num_factors = f->count();

        T* b_ptr = b.data().get();

        launch_kernel_compute_b(f, b_ptr, hessian_ids, jacs, num_factors, std::make_index_sequence<num_vertices>{});
        cudaDeviceSynchronize();

    }

    template<typename F, typename... VertexTypes>
    void compute_Jv(F* f, T* out, T* in) {
        constexpr auto num_vertices = F::get_num_vertices();
        constexpr auto vertex_sizes = F::get_vertex_sizes();

        // std::array<T*, num_vertices> verts;
        auto verts = f->get_vertices();
        std::array<T*, num_vertices> jacs;
        std::array<const size_t*, num_vertices> hessian_ids;
        for (int i = 0; i < num_vertices; i++) {
            // verts[i] = f->vertex_descriptors[i]->x();
            jacs[i] = f->jacobians[i].data.data().get();
            hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
        }

        
        const auto num_factors = f->count();

        launch_kernel_compute_Jv(f, out, in, hessian_ids, jacs, num_factors, std::make_index_sequence<num_vertices>{});
        cudaDeviceSynchronize();

    }


    template<typename F, typename... VertexTypes>
    void compute_Jtv(F* f, T* out, T* in) {
        constexpr auto num_vertices = f->get_num_vertices();
        constexpr auto vertex_sizes = F::get_vertex_sizes();

        // std::array<T*, num_vertices> verts;
        auto verts = f->get_vertices();
        std::array<T*, num_vertices> jacs;
        std::array<const size_t*, num_vertices> hessian_ids;
        for (int i = 0; i < num_vertices; i++) {
            // verts[i] = f->vertex_descriptors[i]->x();
            jacs[i] = f->jacobians[i].data.data().get();
            hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
        }
        
        const auto num_factors = f->count();

        launch_kernel_compute_JtPv(f, out, in, hessian_ids, jacs, num_factors, std::make_index_sequence<num_vertices>{});
        cudaDeviceSynchronize();

    }

    template<typename F>
    void compute_block_diagonal(F* f, std::array<T*,F::get_num_vertices()>& diagonal_blocks) {

        // Then for each vertex, we need to compute the error
        constexpr auto num_vertices = F::get_num_vertices();
        constexpr auto vertex_sizes = F::get_vertex_sizes();

        // At this point all necessary data should be on the GPU    
        // std::array<T*, num_vertices> verts;
        auto verts = f->get_vertices();
        std::array<T*, num_vertices> jacs;
        std::array<const size_t*, num_vertices> hessian_ids;
        for (int i = 0; i < num_vertices; i++) {
            jacs[i] = f->jacobians[i].data.data().get();
            hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
        }

        
        const auto num_factors = f->count();

        launch_kernel_block_diagonal(f, diagonal_blocks, hessian_ids, 
            jacs, num_factors, std::make_index_sequence<num_vertices>{});
        cudaDeviceSynchronize();

    }

    template<typename V>
    void apply_step(V* v) {
        const size_t num_parameters =  v->count()*v->dimension();
        const size_t num_threads = v->count();
        const auto threads_per_block = 256;
        const auto num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

        apply_update_kernel<T, V, typename V::VertexType><<<num_blocks, threads_per_block>>>(
            v->vertices(),
            delta_x.data().get(),
            v->get_hessian_ids(),
            thrust::raw_pointer_cast(v->fixed_mask.data()),
            num_threads
        );
        // cudaDeviceSynchronize();

        // return num_parameters;
    }

    template<typename V>
    void invert_augmented_block_diagonal(V* v, T* block_diagonal, T mu) {
        const size_t num_threads = v->count();
        const auto threads_per_block = 256;
        const auto num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

        invert_hessian_diagonal_kernel<T, V::dim><<<num_blocks, threads_per_block>>>(
            block_diagonal,
            mu,
            thrust::raw_pointer_cast(v->fixed_mask.data()),
            num_threads
        );
        // cudaDeviceSynchronize();

    }

    template<typename V>
    void apply_block_jacobi(V* v, T *z, const T *r, T* block_diagonal) {
        const size_t num_parameters =  v->count()*v->dimension();
        const size_t num_threads = num_parameters;
        const auto threads_per_block = 256;
        const auto num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

        apply_block_jacobi_kernel<T, V::dim><<<num_blocks, threads_per_block>>>(
            z,
            r,
            block_diagonal,
            v->get_hessian_ids(),
            thrust::raw_pointer_cast(v->fixed_mask.data()),
            num_threads
        );
        // cudaDeviceSynchronize();
    }

};
}