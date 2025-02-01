#pragma once
#include <array>
#include <vector>
#include <utility>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <memory>
#include <tuple>

#include <glso/dual.hpp>

// injection for hashing std::pair<size_t, size_t>
namespace std {
    template <>
    struct hash<std::pair<size_t, size_t>> {
        size_t operator()(const std::pair<size_t, size_t>& p) const {
            size_t seed = 0;
            boost::hash_combine(seed, p.first);
            boost::hash_combine(seed, p.second);
            return seed;
            // return std::hash<size_t>()(p.first) ^ std::hash<size_t>()(p.second);
        }
    };
}

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
            // *dst++ = *src++;
            dst[i] = src[i];
        }
    }

template<typename T, size_t N, size_t M, size_t E, typename F, std::size_t... Is>
__global__
void compute_error_kernel(const T* obs, T* error, size_t* ids, std::array<T*, sizeof...(Is)> args, std::index_sequence<Is...>) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr auto vertex_sizes = F::get_vertex_sizes();

    #pragma unroll
    for (int i = 0; i < sizeof...(Is); i++) {
        size_t local_id = ids[idx*N+i];
        args[i] += local_id*vertex_sizes[i];
    }
    
    T local_obs[M];
    T local_error[E];

    #pragma unroll
    for (int i = 0; i < M; ++i) {
        local_obs[i] = obs[idx * M + i];
    }

    #pragma unroll
    for (int i = 0; i < E; ++i) {
        local_error[i] = error[idx * E + i];
    }


    auto v = std::make_tuple(std::array<T, vertex_sizes[Is]>{}...);
    
    auto copy_vertices = [&v, &vertex_sizes](auto&&... ptrs) {
        ((device_copy<T, vertex_sizes[Is]>(ptrs, std::get<Is>(v).data())), ...);
    };

    std::apply(copy_vertices, args);

    F::error(std::get<Is>(v).data()..., local_obs, local_error);

    #pragma unroll
    for(int i = 0; i < E; ++i) {
        error[idx * E + i] = local_error[i];
    }
}

template<typename T>
class GraphVisitor {
public:
    template<typename F, typename... VertexTypes>
    void compute_error(F* f) {
        // Assume autodiff

        // Then for each vertex, we need to compute the error
        constexpr auto num_vertices = f->get_num_vertices();
        

        // At this point all necessary data should be on the GPU    
        std::array<T*, num_vertices> verts;
        // std::array<size_t, num_vertices> stride;
        for (int i = 0; i < num_vertices; i++) {
            verts[i] = f->vertex_descriptors[i]->x();
            // stride[i] = f->vertex_descriptors[i]->dimension();
        }

        
        constexpr auto observation_dim = F::observation_dim;
        constexpr auto error_dim = F::error_dim;
        const auto num_factors = f->count();

        int threads_per_block = 256;
        int num_blocks = (num_factors + threads_per_block - 1) / threads_per_block;

        compute_error_kernel<T, num_vertices, observation_dim, error_dim, F><<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(f->device_obs.data()), 
        f->residuals.data().get(), 
        f->device_ids.data().get(),
        verts, std::make_index_sequence<num_vertices>{});
    }

    template<typename V>
    void apply_step() {
        V::update(nullptr, nullptr);
    }
};

template <typename T>
class BaseVertexDescriptor {
public:
    virtual ~BaseVertexDescriptor() {};

    // virtual void update(const T* x, const T* delta) = 0;
    virtual void visit_update(GraphVisitor<T>& visitor) = 0;
    virtual size_t dimension() const = 0;
    virtual size_t count() const = 0;
    virtual T* x() = 0;
    virtual void to_device() = 0;
    virtual void to_host() = 0;
    virtual void add_vertex(const size_t id, const T* value) = 0;
    virtual const std::unordered_map<size_t, size_t> & get_global_map() const = 0;

};

template <typename T, int D, template <typename> class Derived>
class VertexDescriptor : public BaseVertexDescriptor<T> {
private:
    // Vertex values
    thrust::device_vector<T> x_device;
    thrust::host_vector<T> x_host;

public:

    // Mappings
    std::unordered_map<size_t, size_t> global_to_local_map;

    // using VertexType = V;
    static constexpr size_t dim = D;

public:
    virtual ~VertexDescriptor() {};
    
    void visit_update(GraphVisitor<T>& visitor) override {
        visitor.template apply_step<Derived<T>>();
    }

    virtual void to_device() override {
        x_device = x_host;
    }

    virtual void to_host() override {
        x_host = x_device;
    }

    virtual T* x() override {
        return x_device.data().get();
    }

    virtual size_t count() const override {
        // TODO: Find a better way to get the dimension
        const auto dim = dynamic_cast<const Derived<T>*>(this)->dimension();
        return x_device.size()/dim;
    }

    void add_vertex(const size_t id, const T* value) override {
        // TODO: Find a better way to get the dimension
        const auto dim = dynamic_cast<Derived<T>*>(this)->dimension();        
        x_host.insert(x_host.end(), value, value+dim);
        global_to_local_map.insert({id, (x_host.size()/dim) - 1});
    }

    void reserve(size_t num_vertices) {
        // TODO: Find a better way to get the dimension
        const auto dim = dynamic_cast<Derived<T>*>(this)->dimension();
        x_host.reserve(num_vertices*dim);
        x_device.reserve(num_vertices*dim);
    }

    const std::unordered_map<size_t, size_t> & get_global_map() const override {
        return global_to_local_map;
    }

    size_t dimension() const override {
        return D;
    }

    
};

template<typename T>
class JacobianStorage {
public:

std::pair<size_t, size_t> dimensions;

thrust::device_vector<T> data;


};

// class JacobianInfo {
// public:
// };

template <typename T>
class BaseFactorDescriptor {
public:
    virtual ~BaseFactorDescriptor() {};

    // virtual void error_func(const T** vertices, const T* obs, T* error) = 0;
    virtual bool use_autodiff() = 0;
    virtual void visit_error(GraphVisitor<T>& visitor) = 0;
    virtual std::vector<JacobianStorage<T>> & get_jacobians() = 0;
    // virtual size_t get_num_vertices() const = 0;

    virtual size_t count() const = 0;

    virtual void to_device() = 0;

};

template <typename T, int N, int M, template <typename> class Derived, typename... VertexTypes>
class FactorDescriptor : public BaseFactorDescriptor<T> {

private:
    std::vector<size_t> global_ids;
    thrust::host_vector<size_t> host_ids; // local ids
    thrust::host_vector<T> host_obs;


    


public:

    thrust::device_vector<size_t> device_ids;
    thrust::device_vector<T> device_obs;
    thrust::device_vector<T> residuals;
    std::array<BaseVertexDescriptor<T>*, N> vertex_descriptors;

    static constexpr size_t observation_dim = M;
    static constexpr size_t error_dim = N;
    using VertexTypesTuple = std::tuple<VertexTypes...>;
    VertexTypesTuple vtypes;

    void visit_error(GraphVisitor<T>& visitor) override {
        visitor.template compute_error<Derived<T>, VertexTypes...>(dynamic_cast<Derived<T>*>(this));
    }

    std::vector<JacobianStorage<T>> jacobians;
    
    static constexpr size_t get_num_vertices() {
        return N;
    }

    std::vector<JacobianStorage<T>> &  get_jacobians() override {
        return jacobians;
    }

    void add_factor(const std::array<size_t, N>& ids, const std::array<T, M>& obs, const T* precision_matrix) {
        
        global_ids.insert(global_ids.end(), ids.begin(), ids.end());
        host_obs.insert(host_obs.end(), obs.begin(), obs.end());

    }

    size_t count() const override {
        return device_ids.size()/N;
    }

    void to_device() override {

        // Map constraint ids to local ids
        for (size_t i = 0; i < global_ids.size(); i++) {
            host_ids.push_back(vertex_descriptors[i%N]->get_global_map().at(global_ids[i]));
        }

        device_ids = host_ids;
        device_obs = host_obs;
    }

    void link_factors(const std::array<BaseVertexDescriptor<T>*, N>& vertex_descriptors) {
        this->vertex_descriptors = vertex_descriptors;
    }

    static constexpr std::array<size_t, N> get_vertex_sizes() {
        return {VertexTypes::dim...};
    }

};

// Templated derived class for AutoDiffFactorDescriptor using CRTP
// N is the number of vertices involved in the constraint
// M is the dimension of each observation
template <typename T, int N, int M, template <typename> class Derived, typename... VertexTypes>
class AutoDiffFactorDescriptor : public FactorDescriptor<T, N, M, Derived, VertexTypes...> {
public:
    virtual bool use_autodiff() override {
        return true;
    }
};



class Vertex {
private:
    size_t id;

public:
    void set_id(size_t id) {
        this->id = id;
    }
};



template<typename T=double>
class Graph {

    private:

    GraphVisitor<T> visitor;

    std::vector<BaseVertexDescriptor<T>*> vertex_descriptors;
    std::vector<BaseFactorDescriptor<T>*> factor_descriptors;

    // std::unordered_map<std::pair<size_t, size_t>, std::shared_ptr<JacobianStorage<T>>> jacobians;
    std::unordered_map<size_t, std::pair<size_t, size_t>> hessian_to_local_map;
    std::unordered_map<size_t, size_t> hessian_offset;
    // Solver buffers
    thrust::device_vector<T> delta_x;
    thrust::device_vector<T> b;
    thrust::device_vector<T> x_backup;

    public:

    void add_vertex_descriptor(BaseVertexDescriptor<T>* descriptor) {
        vertex_descriptors.push_back(descriptor);
    }

    template<typename F, typename ... Ts>
    F* add_factor_descriptor(Ts ... vertices) {

        // Link factor to vertices
        auto factor = new F();
        factor->link_factors({vertices...});

        factor_descriptors.push_back(factor);

        // Create Jacobian storage for this factor
        // const auto & info_list = descriptor->get_jacobian_info();

        // for (const auto & info: info_list) {
        //     jacobians.insert({info.dim, std::make_shared<JacobianStorage<T>>(info.nnz())});
        // }

        return factor;

    }

    bool initialize_optimization() {

        // For each vertex descriptor, take global to local id mapping and transform it into a Hessian 
        // column to local id mapping.
        
        std::vector<std::pair<size_t, std::pair<size_t, size_t>>> global_to_local_combined;

        for (size_t i = 0; i < vertex_descriptors.size(); ++i) {
            const auto& map = vertex_descriptors[i]->get_global_map();
            for (const auto& entry : map) {
            global_to_local_combined.push_back({entry.first, {i, entry.second}});
            }
        }
        
        // Sort the combined list by global ID
        std::sort(global_to_local_combined.begin(), global_to_local_combined.end(), 
            [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // Assign Hessian columns to local indices
        hessian_to_local_map.clear();
        hessian_offset.clear();
        size_t hessian_column = 0;
        size_t offset = 0;
        for (const auto& entry : global_to_local_combined) {
            hessian_to_local_map.insert({hessian_column, entry.second});
            hessian_offset.insert({hessian_column, offset});
            offset += vertex_descriptors[entry.second.first]->dimension();
            hessian_column++;
        }

        // Transform global vertex ids into local ids for factors
        // for (const auto & [global_id, second]: global_to_local_combined) {
        //     const auto & [vd_idx, local_id] = second;
        
        // }

        // Copy vertex values to device
        for (auto & desc: vertex_descriptors) {
            desc->to_device();
        }

        // Copy factors to device
        for (auto & desc: factor_descriptors) {
            desc->to_device();
        }

        return true;
    }

    bool build_structure() {
        // Allocate storage for Jacobians
        // Allocate storage for solver vectors
        size_t size_x = 0;
        for (const auto & desc: vertex_descriptors) {
            size_x += desc->dimension()*desc->count(); 
        }

        delta_x.resize(size_x);
        b.resize(size_x);
        x_backup.resize(size_x);

        return false;
    }

    void linearize() {
        // clear delta_x and b
        thrust::fill(delta_x.begin(), delta_x.end(), 0);
        thrust::fill(b.begin(), b.end(), 0);


        for (auto & factor: factor_descriptors) {
            // compute error
            factor->visit_error(visitor);
            if (factor->use_autodiff()) {
                // copy gradients into Jacobians
            }
            else {
                // compute Jacobians
            }
        }
    }

    bool compute_step() {

        return false;
    }

    void apply_step() {
        for (auto & desc: vertex_descriptors) {
            desc->visit_update(visitor);
        }
    }

    void backup_parameters() {
        size_t offset = 0;
        for (const auto & desc: vertex_descriptors) {
            const size_t param_size = desc->dimension()*desc->count();
            const T* x = desc->x();
            thrust::copy(x, x+param_size, x_backup.begin()+offset);
            offset += param_size;
        }

    }

    void revert_parameters() {
        size_t offset = 0;
        for (auto & desc: vertex_descriptors) {
            const size_t param_size = desc->dimension()*desc->count();
            T* x = desc->x();
            thrust::copy(x_backup.begin()+offset, x_backup.begin(), x);
            offset += param_size;
        }

    }

};



template<typename T=double>
class Optimizer {
public:
    bool optimize(Graph<T>* graph, const size_t num_iterations) {

        // Initialize something for all iterations

        if (!graph->initialize_optimization()) {
            return false;
        }

        if (!graph->build_structure()) {
            return false;
        }

        for (size_t i = 0; i < num_iterations; i++) {

            graph->linearize();

            if(!graph->compute_step()) {
                return false;
            }

            graph->backup_parameters();
            graph->apply_step();

            // Try step
            bool step_is_good = true; // make this a real check later

            if (step_is_good) {
                // update hyperparameters
            }
            else {
                graph->revert_parameters();
                // update hyperparameters
            }
        }
        return true;
    }

};

}


