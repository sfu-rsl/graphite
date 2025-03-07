#pragma once
#include <glso/graph.hpp>

namespace glso {
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