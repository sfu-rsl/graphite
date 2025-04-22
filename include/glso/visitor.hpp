#pragma once
#include <glso/common.hpp>

namespace glso {

    template <typename T>
    __device__ void device_copy(const T* src, const T* src_end, T* dst) {
        while (src != src_end) {
            *dst++ = *src++;
        }
    }

    template <typename T, size_t D>
    __device__ void device_copy(const T* src, T* dst) {
        #pragma unroll
        for (size_t i = 0; i < D; i++) {
            dst[i] = src[i];
        }
    }

    template <typename T, size_t D>
    __device__ void real_to_dual(const T* src, Dual<T>* dst) {
        #pragma unroll
        for (size_t i = 0; i < D; i++) {
            dst[i] = Dual<T>(src[i]);
        }
    }

template<typename T, typename V>
__global__ void apply_update_kernel(T* x, const T* delta_x, const size_t * hessian_ids, const size_t num_threads) {
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex_id >= num_threads) {
        return;
    }

    x += vertex_id * V::dim;
    const T* delta = delta_x + hessian_ids[vertex_id];

    V::update(x, delta);

}

template<typename T, size_t I, size_t N, size_t M, size_t E, typename F, std::size_t... Is>
__global__
void compute_error_kernel_autodiff(const T* obs, T* error, size_t* ids, const size_t* hessian_ids, const size_t num_threads, std::array<T*, sizeof...(Is)> args, std::array<T*, sizeof...(Is)> jacs, std::index_sequence<Is...>) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_threads) {
        return;
    }


    constexpr auto vertex_sizes = F::get_vertex_sizes();
    const auto factor_id = idx / vertex_sizes[I];
    const auto vertex_id = ids[factor_id*N + I];

    // printf("CEAD: Thread %d, Vertex %d, Factor %d\n", idx, vertex_id, factor_id);

    #pragma unroll
    for (int i = 0; i < sizeof...(Is); i++) {
        size_t local_id = ids[factor_id*N+i];
        args[i] += local_id*vertex_sizes[i];
    }
    
    Dual<T> local_obs[M];
    Dual<T> local_error[E];

    #pragma unroll
    for (int i = 0; i < M; ++i) {
        local_obs[i] = obs[factor_id * M + i];
    }

    #pragma unroll
    for (int i = 0; i < E; ++i) {
        local_error[i] = error[factor_id * E + i];
    }


    auto v = cuda::std::make_tuple(std::array<Dual<T>, vertex_sizes[Is]>{}...);
    
    auto copy_vertices = [&v, &vertex_sizes](auto&&... ptrs) {
        ((real_to_dual<T, vertex_sizes[Is]>(ptrs, cuda::std::get<Is>(v).data())), ...);
    };

    std::apply(copy_vertices, args);

    cuda::std::get<I>(v)[idx % vertex_sizes[I]].dual = static_cast<T>(1);

    F::error(cuda::std::get<Is>(v).data()..., local_obs, local_error);


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
    #pragma unroll
    for(size_t i = 0; i < E; ++i) {
        jacs[I][j_size*factor_id + col_offset + i] = local_error[i].dual;
        // printf("Jacobian[%d] = %f\n", j_size*factor_id + col_offset + i, jacs[I][j_size*factor_id + col_offset + i]);
    }
}
// TODO: Make this more efficient and see if code can be shared with the autodiff kernel
template<typename T, size_t N, size_t M, size_t E, typename F, std::size_t... Is>
__global__
void compute_error_kernel(const T* obs, T* error, size_t* ids, const size_t num_threads, std::array<T*, sizeof...(Is)> args, std::index_sequence<Is...>) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_threads) {
        return;
    }

    constexpr auto vertex_sizes = F::get_vertex_sizes();
    const auto factor_id = idx;

    #pragma unroll
    for (int i = 0; i < sizeof...(Is); i++) {
        size_t local_id = ids[factor_id*N+i];
        args[i] += local_id*vertex_sizes[i];
    }
    
    T local_obs[M];
    T local_error[E];

    #pragma unroll
    for (int i = 0; i < M; ++i) {
        local_obs[i] = obs[factor_id * M + i];
    }

    #pragma unroll
    for (int i = 0; i < E; ++i) {
        local_error[i] = error[factor_id * E + i];
    }


    auto v = cuda::std::make_tuple(std::array<T, vertex_sizes[Is]>{}...);
    
    auto copy_vertices = [&v, &vertex_sizes](auto&&... ptrs) {
        ((device_copy<T, vertex_sizes[Is]>(ptrs, cuda::std::get<Is>(v).data())), ...);
    };

    std::apply(copy_vertices, args);

    F::error(cuda::std::get<Is>(v).data()..., local_obs, local_error);


    #pragma unroll
    for(size_t i = 0; i < E; ++i) {
        error[factor_id * E + i] = local_error[i];
    }

}

// The output will be part of b with length of the vertex (where b = -J^T * r)
// Note the negative sign - different papers use different conventions
// TODO: Replace with generic J^T x r kernel?
// Note: The error vector is local to the factor
template<typename T, size_t I, size_t N, size_t M, size_t E, typename F, std::size_t... Is>
__global__ 
void compute_b_kernel(T* b, T* error, size_t* ids, const size_t* hessian_ids, const size_t num_threads, std::array<T*, sizeof...(Is)> jacs, std::index_sequence<Is...>) {
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

// Compute J * x where the length of vector x matches the Hessian dimension
// Each Jacobian block needs to be accessed just once
// So we need E threads for each block (error dimension)
// In total we should hae E*num_factors threads?
template<typename T, size_t I, size_t N, size_t M, size_t E, typename F, std::size_t... Is>
__global__ 
void compute_Jv_kernel(T*y, T* x, T* error, size_t* ids, const size_t* hessian_ids, const size_t num_threads, std::array<T*, sizeof...(Is)> jacs, std::index_sequence<Is...>) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_threads) {
        return;
    }

    constexpr auto vertex_sizes = F::get_vertex_sizes();
    constexpr auto jacobian_size = vertex_sizes[I]*E;
    
    // Each J block is stored as E x d col major, where d is the vertex size
    const size_t factor_id = idx / E;
    const auto jacobian_offset = factor_id * jacobian_size;

    T value = 0;
    constexpr auto d = vertex_sizes[I];

    size_t local_id = ids[factor_id*N+I]; // N is the number of vertices involved in the factor
    const auto hessian_offset = hessian_ids[local_id]; // each vertex has a hessian_ids array
    const auto row_offset = (idx % E);
    // Adding i*E skips to the next column
    size_t residual_offset = 0; // need to pass this in
    // it's the offset into the r vector
    #pragma unroll
    for (int i = 0; i < d; i++) {
        value += jacs[I][jacobian_offset + row_offset + i*E] * x[hessian_offset + i];
    }
    
    atomicAdd(&y[residual_offset + idx], value);

}

// Compute J^T * x where x is the size of the residual vector
// Each Jacobian block needs to be accessed just once
// For each block, we need d threads where d is the vertex size
// We need to load the x vector location for the corresponding block row of J
// So this assumes that the x vector has the same layout as the residual vector for this factor (rather than a global residual vector)
// The aggregate output will be H x len(x) where H is hessian dimension
template<typename T, size_t I, size_t N, size_t M, size_t E, typename F, std::size_t... Is>
__global__ 
void compute_Jtv_kernel(T* y, T* x, size_t* ids, const size_t* hessian_ids, const size_t num_threads, std::array<T*, sizeof...(Is)> jacs, std::index_sequence<Is...>) {
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
        value += jacs[I][jacobian_offset + col_offset + i] * x[error_offset + i];
    }

    size_t local_id = ids[factor_id*N+I]; // N is the number of vertices involved in the factor
    const auto hessian_offset = hessian_ids[local_id]; // each vertex has a hessian_ids array
    
    atomicAdd(&y[hessian_offset + (idx % vertex_sizes[I])], value);

}

template<typename T>
class GraphVisitor {
private:

thrust::device_vector<T>& delta_x;
thrust::device_vector<T>& b;

template <typename F, std::size_t... Is>
void launch_kernel_autodiff(F* f, std::array<const size_t*, F::get_num_vertices()>& hessian_ids, std::array<T*, F::get_num_vertices()>& verts, std::array<T*, F::get_num_vertices()> & jacs, const size_t num_factors, std::index_sequence<Is...>) {
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

            compute_error_kernel_autodiff<T, Is, num_vertices, F::observation_dim, F::error_dim, F><<<num_blocks, threads_per_block>>>(
                f->device_obs.data().get(),
                f->residuals.data().get(),
                f->device_ids.data().get(),
                hessian_ids[Is],
                num_threads,
                verts,
                jacs,
                std::make_index_sequence<num_vertices>{});
            }()), ...);
        }

template <typename F, std::size_t... Is>
void launch_kernel_compute_b(F* f, T* b, std::array<const size_t*, F::get_num_vertices()>& hessian_ids, std::array<T*, F::get_num_vertices()>& verts, std::array<T*, F::get_num_vertices()> & jacs, const size_t num_factors, std::index_sequence<Is...>) {
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

            compute_b_kernel<T, Is, num_vertices, F::observation_dim, F::error_dim, F><<<num_blocks, threads_per_block>>>(
                b,
                f->residuals.data().get(),
                f->device_ids.data().get(),
                hessian_ids[Is],
                num_threads,
                jacs,
                std::make_index_sequence<num_vertices>{});
            }()), ...);
        }



        template <typename F, std::size_t... Is>
        void launch_kernel_compute_Jtv(F* f, T* out, T* in, std::array<const size_t*, F::get_num_vertices()>& hessian_ids, std::array<T*, F::get_num_vertices()>& verts, std::array<T*, F::get_num_vertices()> & jacs, const size_t num_factors, std::index_sequence<Is...>) {
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
        
                    compute_Jtv_kernel<T, Is, num_vertices, F::observation_dim, F::error_dim, F><<<num_blocks, threads_per_block>>>(
                        out,
                        in,
                        f->device_ids.data().get(),
                        hessian_ids[Is],
                        num_threads,
                        jacs,
                        std::make_index_sequence<num_vertices>{});
                    }()), ...);
                }

        template <typename F, std::size_t... Is>
        void launch_kernel_compute_Jv(F* f, T* out, T* in, std::array<const size_t*, F::get_num_vertices()>& hessian_ids, std::array<T*, F::get_num_vertices()>& verts, std::array<T*, F::get_num_vertices()> & jacs, const size_t num_factors, std::index_sequence<Is...>) {
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
        
                    compute_Jv_kernel<T, Is, num_vertices, F::observation_dim, F::error_dim, F><<<num_blocks, threads_per_block>>>(
                        out,
                        in,
                        f->residuals.data().get(),
                        f->device_ids.data().get(),
                        hessian_ids[Is],
                        num_threads,
                        jacs,
                        std::make_index_sequence<num_vertices>{});
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

        
        constexpr auto observation_dim = F::observation_dim;
        constexpr auto error_dim = F::error_dim;
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

        
        constexpr auto observation_dim = F::observation_dim;
        constexpr auto error_dim = F::error_dim;
        const auto num_factors = f->count();

        const auto num_threads = num_factors;
        int threads_per_block = 256;
        int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

        compute_error_kernel<T, num_vertices, F::observation_dim, F::error_dim, F><<<num_blocks, threads_per_block>>>(
            f->device_obs.data().get(),
            f->residuals.data().get(),
            f->device_ids.data().get(),
            num_threads,
            verts,
            std::make_index_sequence<num_vertices>{});

        // launch_kernel_error(f, hessian_ids, verts, num_factors, std::make_index_sequence<num_vertices>{});
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

        
        constexpr auto observation_dim = F::observation_dim;
        constexpr auto error_dim = F::error_dim;
        const auto num_factors = f->count();

        T* b_ptr = b.data().get();

        launch_kernel_compute_b(f, b_ptr, hessian_ids, verts, jacs, num_factors, std::make_index_sequence<num_vertices>{});
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

        
        constexpr auto observation_dim = F::observation_dim;
        constexpr auto error_dim = F::error_dim;
        const auto num_factors = f->count();

        launch_kernel_compute_Jv(f, out, in, hessian_ids, verts, jacs, num_factors, std::make_index_sequence<num_vertices>{});
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

        
        constexpr auto observation_dim = F::observation_dim;
        constexpr auto error_dim = F::error_dim;
        const auto num_factors = f->count();

        launch_kernel_compute_Jtv(f, out, in, hessian_ids, verts, jacs, num_factors, std::make_index_sequence<num_vertices>{});
        cudaDeviceSynchronize();

    }

    template<typename V>
    void apply_step(V* v) {
        // V::update(nullptr, nullptr);
        const size_t num_parameters =  v->count()*v->dimension();
        const size_t num_threads = v->count();
        const auto threads_per_block = 256;
        const auto num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

        apply_update_kernel<T, V><<<num_blocks, threads_per_block>>>(
            v->vertices(),
            delta_x.data().get(),
            v->get_hessian_ids(),
            num_threads
        );
        // cudaDeviceSynchronize();

        // return num_parameters;
    }
};
}