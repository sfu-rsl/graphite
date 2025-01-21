#pragma once
#include <vector>
#include <utility>
#include <unordered_map>
#include <boost/functional/hash.hpp>


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
        F::error_func();
    }

    template<typename V>
    void apply_step() {
        V::update();
    }
};

template <typename T>
class BaseVertexDescriptor {
public:
    virtual ~BaseVertexDescriptor() {};

    virtual void update(const T* delta) = 0;
    virtual void visit_update(GraphVisitor<T>& visitor) = 0;
};

template <typename T, template <typename> class Derived>
class VertexDescriptor : public BaseVertexDescriptor<T> {
public:
    virtual ~VertexDescriptor() {};
    
    void visit_update(GraphVisitor<T>& visitor) override {
        visitor.template apply_step<Derived<T>>();
    }
};

template <typename T>
class BaseFactorDescriptor {
public:
    virtual ~BaseFactorDescriptor() {};

    virtual void error_func(const T** vertices, const T* obs, T* error) = 0;
    virtual bool use_autodiff() = 0;
    virtual void visit_error(GraphVisitor<T>& visitor) = 0;

};

template <typename T, template <typename> class Derived>
class FactorDescriptor : public BaseFactorDescriptor<T> {
public:

    void visit_error(GraphVisitor<T>& visitor) override {
        visitor.template compute_error<Derived<T>>();
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

class JacobianStorage {
public:

};

template<typename T=double>
class Graph {

    private:

    std::vector<BaseVertexDescriptor<T>*> vertex_descriptors;
    std::vector<BaseFactorDescriptor<T>*> factor_descriptors;

    std::unordered_map<std::pair<size_t, size_t>, JacobianStorage> jacobians;

    public:

    void add_vertex_descriptor(BaseVertexDescriptor<T>* descriptor) {
        vertex_descriptors.push_back(descriptor);
    }

    void add_factor_descriptor(BaseFactorDescriptor<T>* descriptor) {
        factor_descriptors.push_back(descriptor);
    }

    bool build_structure() {

        return false;
    }

    void linearize() {
        for (auto & factor: factor_descriptors) {
            // compute error
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

    }

    void backup_parameters() {

    }

    void revert_parameters() {

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


