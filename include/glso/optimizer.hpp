#pragma once
#include <glso/graph.hpp>
#include <iomanip>

namespace glso {
template<typename T=double>
class Optimizer {

private:


T compute_rho(Graph<T>* graph, const T chi2, const T new_chi2, const T mu) {
    // Compute rho
    //  TODO: Don't store these in the graph
    auto & delta_x = graph->get_delta_x();
    auto & b = graph->get_b();
    T num = (chi2 - new_chi2);
    T denom = (mu * thrust::inner_product(delta_x.begin(), delta_x.end(), delta_x.begin(), 0.0));
    denom += thrust::inner_product(delta_x.begin(), delta_x.end(), b.begin(), 0.0);
    return num / (denom+1e-10);
}

public:

    // TODO: Remove the damping factor from Graph
    bool optimize(Graph<T>* graph, const size_t num_iterations, T damping_factor=1e-2) {

        // Initialize something for all iterations
        auto start = std::chrono::steady_clock::now();
        T mu = damping_factor;
        T nu = 2;

        if (!graph->initialize_optimization()) {
            return false;
        }

        if (!graph->build_structure()) {
            return false;
        }

        graph->set_damping_factor(mu);
        graph->linearize();

        bool run  = true;

        double time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        // Print iteration table headers
        std::cout << std::setw(10) << "Iteration" 
              << std::setw(16) << "Initial Chi2" 
              << std::setw(16) << "Current Chi2" 
              << std::setw(16) << "Lambda" 
              << std::setw(16) << "Time" 
              << std::setw(16) << "Total Time" 
              << std::endl;
        std::cout << "------------------------------------------------------------------------------------------" << std::endl;

        for (size_t i = 0; i < num_iterations && run; i++) {
            start = std::chrono::steady_clock::now();
            T chi2 = graph->chi2();

            if(!graph->compute_step()) {
                return false;
            }

            graph->backup_parameters();
            graph->apply_step();

            // Try step
            graph->compute_error();
            T new_chi2 = graph->chi2();

            // std::cout << "Iteration " << i << ", chi2: " << chi2 << ", candidate chi2: " << new_chi2 << std::endl;
            bool step_is_good = std::isfinite(new_chi2);

            T rho = compute_rho(graph, chi2, new_chi2, mu);

            if (step_is_good && rho > 0) {
                // update hyperparameters
                mu *= std::max(std::max(static_cast<T>(1.0/3.0), 1 - pow(2*rho-1, 3)), 1e-12);
                nu = 2;
                // Relinearize since step is accepted
                graph->set_damping_factor(mu);
                graph->linearize();
                // std::cout << "Good step" << std::endl;
            }
            else {
                graph->revert_parameters();
                graph->compute_error();
                // update hyperparameters
                mu *= nu;
                nu *= 2;
                graph->set_damping_factor(mu);
                // std::cout << "Bad step" << std::endl;
                new_chi2 = chi2;
            }

            double iteration_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            time += iteration_time;
            std::cout << std::setw(10) << i 
                      << std::setw(16) << chi2 
                      << std::setw(16) << new_chi2 
                      << std::setw(16) << mu 
                      << std::setw(16) << iteration_time
                      << std::setw(16) << time
                      << std::endl;


            if (!std::isfinite(mu)) {
                std::cout << "Damping factor is infinite, terminating optimization" << std::endl;
                run = false;
            }

        }

        // Should only really do this when optimization is successful
        graph->to_host();

        return run;
    }

};
}