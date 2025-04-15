#pragma once
#include <glso/common.hpp>
#include <glso/visitor.hpp>
namespace glso {

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
    virtual const size_t* get_hessian_ids() const = 0;
    virtual void set_hessian_column(size_t global_id, size_t hessian_column) = 0;

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
    thrust::host_vector<size_t> local_to_hessian_offsets;
    thrust::device_vector<size_t> hessian_ids;

    static constexpr size_t dim = D;

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
        local_to_hessian_offsets.push_back(0); // Initialize to 0
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
        return D;
    }

    const size_t* get_hessian_ids() const override {
        return hessian_ids.data().get();
    }

    void set_hessian_column(size_t global_id, size_t hessian_column) {
        local_to_hessian_offsets[global_to_local_map.at(global_id)] = hessian_column;
    }

};

}