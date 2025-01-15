#pragma once
#include <vector>

namespace glso {

template <typename T>
class VertexDescriptor {
public:
    virtual ~VertexDescriptor() {};
};

template <typename T>
class FactorDescriptor {
public:
    virtual ~FactorDescriptor() {};

    virtual void error_func(const T** vertices, const T* obs, T* error) = 0;
    virtual bool use_autodiff() = 0;
};

template<typename T>
class GraphVisitor {
public:
    template<typename F>
    void compute_error() {
        // Do something with the factor's error function
        F::error_func();
    }
};

// Templated derived class for AutoDiffFactorDescriptor using CRTP
template <typename T, template <typename> class Derived>
class AutoDiffFactorDescriptor : public FactorDescriptor<T> {
public:
    virtual bool use_autodiff() override {
        return true;
    }

    void visit_error(GraphVisitor<T>& visitor) {
        visitor.template compute_error<Derived<T>>();
    }
};

template <typename T>
class TestFactor : public AutoDiffFactorDescriptor<T, TestFactor> {
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

    std::vector<VertexDescriptor<T>*> vertex_descriptors;
    std::vector<FactorDescriptor<T>*> factor_descriptors;

    public:

    void add_vertex_descriptor(VertexDescriptor<T>* descriptor) {
        vertex_descriptors.push_back(descriptor);
    }

    void add_factor_descriptor(FactorDescriptor<T>* descriptor) {
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


