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

        bool run  = true;

        for (size_t i = 0; i < num_iterations && run; i++) {

            graph->linearize();

            T chi2 = graph->chi2();

            if(!graph->compute_step()) {
                return false;
            }

            graph->backup_parameters();
            graph->apply_step();

            // Try step
            graph->compute_error();
            T new_chi2 = graph->chi2();

            std::cout << "Iteration " << i << ", chi2: " << chi2 << ", new chi2: " << new_chi2 << std::endl;
            bool step_is_good = new_chi2 <= chi2;

            if (step_is_good) {
                // update hyperparameters
                // std::cout << "Good step, chi2: " << new_chi2 << std::endl;
            }
            else {
                graph->revert_parameters();
                // update hyperparameters
                // std::cout << "Bad step, aborting" << std::endl;
                run = false;
            }
        }
        return run;
    }

};
}