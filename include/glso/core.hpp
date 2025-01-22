#pragma once
#include <vector>
#include <utility>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <memory>


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


template<typename T>
class GraphVisitor {
public:
    template<typename F>
    void compute_error() {
        // Do something with the factor's error function
        F::error_func(nullptr, nullptr, nullptr);
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

};

template <typename T, template <typename> class Derived>
class VertexDescriptor : public BaseVertexDescriptor<T> {
private:
    // Vertex values
    thrust::device_vector<T> x_device;
    thrust::host_vector<T> x_host;

    // Mappings
    std::unordered_map<size_t, size_t> idx_map;

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
        idx_map.insert({id, (x_host.size()/dim) - 1});
    }

    void reserve(size_t num_vertices) {
        // TODO: Find a better way to get the dimension
        const auto dim = dynamic_cast<Derived<T>*>(this)->dimension();
        x_host.reserve(num_vertices*dim);
        x_device.reserve(num_vertices*dim);
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

};

template <typename T, template <typename> class Derived>
class FactorDescriptor : public BaseFactorDescriptor<T> {
public:

    void visit_error(GraphVisitor<T>& visitor) override {
        visitor.template compute_error<Derived<T>>();
    }

    std::vector<JacobianStorage<T>> jacobians;

    std::vector<JacobianStorage<T>> &  get_jacobians() override {
        return jacobians;
    }

};

// Templated derived class for AutoDiffFactorDescriptor using CRTP
template <typename T, template <typename> class Derived>
class AutoDiffFactorDescriptor : public FactorDescriptor<T, Derived> {
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

    // Solver buffers
    thrust::device_vector<T> delta_x;
    thrust::device_vector<T> b;
    thrust::device_vector<T> x_backup;

    public:

    void add_vertex_descriptor(BaseVertexDescriptor<T>* descriptor) {
        vertex_descriptors.push_back(descriptor);
    }

    void add_factor_descriptor(BaseFactorDescriptor<T>* descriptor) {
        factor_descriptors.push_back(descriptor);

        // Create Jacobian storage for this factor
        // const auto & info_list = descriptor->get_jacobian_info();

        // for (const auto & info: info_list) {
        //     jacobians.insert({info.dim, std::make_shared<JacobianStorage<T>>(info.nnz())});
        // }

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


