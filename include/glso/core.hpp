#pragma once

namespace glso {


template <typename T, int D>
class VertexDescriptor {
private:


public:

void update(T* vertex, const T* update) const;


};

// V is the number of vertices
// O is the dimension of the observation
// E is the dimension of the error vector
template <typename T, int V, int O, int E>
class FactorDescriptor {
public:

// T** get_vertices(Graph* graph);
// const get_vertex_types() 

void compute_error(const T** vertices, const T* obs, T* error) const;

};


class Vertex {
    private:

    size_t id;

    public:

    void set_id(size_t id) {
        this->id = id;
    }

};

class Factor {


};

class Graph {

    public:

    void add_vertex_descriptor() {
        
    }

    bool build_structure() {

        return false;
    }

    void linearize() {

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

class Optimizer {
public:
    bool optimize(Graph* graph, const size_t num_iterations) {

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


