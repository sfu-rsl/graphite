#pragma once
#include <glso/common.hpp>
#include <glso/visitor.hpp>
namespace glso {

// T is the float type and D is the dimension of the parameterization
template <typename T, int D>
class BaseVertex {
public:
    virtual ~BaseVertex() {};

    __host__ __device__ virtual void update(const T* delta) = 0;
    __host__ __device__ virtual std::array<T, D> params() const = 0;

    static constexpr int dimension = D;

};

template <typename VertexType, typename T>
__global__ void backup_parameters_kernel(const VertexType* vertices, T* dst, const size_t num_vertices) {
        
    const size_t vertex_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex_id >= num_vertices) return;
    
    const auto params = vertices[vertex_id].params();

    const size_t dst_offset = vertex_id * VertexType::dimension;

    for (size_t i = 0; i < VertexType::dimension; ++i) {
        dst[dst_offset + i] = params[i];
    }
}

template <typename VertexType, typename T>
__global__ void set_parameters_kernel(VertexType* vertices, const T* src, const size_t num_vertices) {
        
    const size_t vertex_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex_id >= num_vertices) return;
    
    const size_t src_offset = vertex_id * VertexType::dimension;

    for (size_t i = 0; i < VertexType::dimension; ++i) {
        // vertices[vertex_id].update(src[src_offset + i]);
    }
}

template <typename T>
class BaseVertexDescriptor {
public:
    virtual ~BaseVertexDescriptor() {};

    // virtual void update(const T* x, const T* delta) = 0;
    virtual void visit_update(GraphVisitor<T>& visitor) = 0;
    virtual size_t dimension() const = 0;
    virtual size_t count() const = 0;
    // virtual T* x() = 0;
    virtual void get_parameters(T* dst) const = 0;
    virtual void set_parameters(const T* src) = 0;
    virtual void to_device() = 0;
    virtual void to_host() = 0;
    // virtual void add_vertex(const size_t id, const BaseVertex<T> & vertex) = 0;
    virtual const std::unordered_map<size_t, size_t> & get_global_map() const = 0;
    virtual const size_t* get_hessian_ids() const = 0;
    virtual void set_hessian_column(size_t global_id, size_t hessian_column) = 0;

};

template <typename T, typename V, template <typename> class Derived>
class VertexDescriptor : public BaseVertexDescriptor<T> {
public:
// using VertexType = typename Derived<T>::VertexType;
using VertexType = V;
    
private:
    // Vertex values
    // thrust::device_vector<T> x_device;
    // thrust::host_vector<T> x_host;

    thrust::device_vector<VertexType> x_device;
    thrust::host_vector<VertexType> x_host;

public:

    // Mappings
    std::unordered_map<size_t, size_t> global_to_local_map;
    thrust::host_vector<size_t> local_to_hessian_offsets;
    thrust::device_vector<size_t> hessian_ids;

    // static constexpr size_t dim = D;
    static constexpr size_t dim = V::dimension;

public:
    virtual ~VertexDescriptor() {};
    
    void visit_update(GraphVisitor<T>& visitor) override {
        visitor.template apply_step<Derived<T>>(dynamic_cast<Derived<T>*>(this));
    }

    virtual void to_device() override {
        x_device = x_host;
        hessian_ids = local_to_hessian_offsets;
    }

    virtual void to_host() override {
        x_host = x_device;
    }

    // virtual T* x() override {
    //     return x_device.data().get();
    // }

    VertexType* vertices() {
        return x_device.data().get();
    }



    virtual void get_parameters(T* dst) const override {
        // const size_t param_size = desc->dimension()*desc->count();
        const VertexType* vertices = x_device.data().get();

        const int num_vertices = static_cast<int>(count());
        const int num_threads = num_vertices;
        const int block_size = 256;
        const auto num_blocks = (num_threads + block_size - 1) / block_size;
        backup_parameters_kernel<VertexType, T><<<num_blocks, block_size>>>(vertices, dst, num_vertices);
    }

    // __global__ static void set_parameters_kernel(VertexType* vertices, const T* src, const size_t num_vertices) {
        
    //     const size_t vertex_id = blockIdx.x * blockDim.x + threadIdx.x;

    //     if (vertex_id >= num_vertices) return;
        
    //     const size_t src_offset = vertex_id * VertexType::dimension;

    //     for (size_t i = 0; i < VertexType::dimension; ++i) {
    //         vertices[vertex_id].update(src[src_offset + i]);
    //     }
    // }

    virtual void set_parameters(const T* src) override {
        // const size_t param_size = desc->dimension()*desc->count();
        VertexType* vertices = x_device.data().get();

        const int num_vertices = static_cast<int>(count());
        const int num_threads = num_vertices;
        const int block_size = 256;
        const auto num_blocks = (num_threads + block_size - 1) / block_size;
        set_parameters_kernel<VertexType, T><<<num_blocks, block_size>>>(vertices, src, num_vertices);
    }

    

    virtual size_t count() const override {
        // TODO: Find a better way to get the dimension
        // const auto dim = dynamic_cast<const Derived<T>*>(this)->dimension();
        // return x_device.size()/dim;
        return x_device.size();
    }

    void add_vertex(const size_t id, const BaseVertex<T, dim>& vertex) {
        // TODO: Find a better way to get the dimension
        // const auto dim = dynamic_cast<Derived<T>*>(this)->dimension();        
        // x_host.insert(x_host.end(), value, value+dim);
        x_host.push_back(static_cast<const VertexType&>(vertex));
        global_to_local_map.insert({id, (x_host.size()) - 1});
        local_to_hessian_offsets.push_back(0); // Initialize to 0
    }

    VertexType get_vertex(const size_t id) {
        // std::array<T, D> vertex_data;
        const auto local_id = global_to_local_map.at(id);
        return x_host[local_id];
        // const T* data_ptr = x_host.data() + local_id * D;
        // std::copy(data_ptr, data_ptr + D, vertex_data.begin());
        // return vertex_data;
    }

    // void reserve(size_t num_vertices) {
    //     // TODO: Find a better way to get the dimension
    //     const auto dim = dynamic_cast<Derived<T>*>(this)->dimension();
    //     x_host.reserve(num_vertices*dim);
    //     x_device.reserve(num_vertices*dim);
    // }

    const std::unordered_map<size_t, size_t> & get_global_map() const override {
        return global_to_local_map;
    }

    size_t dimension() const override {
        return dim;
        // return Derived<T>::VertexType::dimension;
    }

    const size_t* get_hessian_ids() const override {
        return hessian_ids.data().get();
    }

    void set_hessian_column(size_t global_id, size_t hessian_column) {
        local_to_hessian_offsets[global_to_local_map.at(global_id)] = hessian_column;
    }

};

}