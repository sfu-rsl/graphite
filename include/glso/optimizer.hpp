#pragma once
#include <glso/graph.hpp>
#include <glso/solver.hpp>
#include <iomanip>

namespace glso {

namespace optimizer {

template <typename T, typename S>
T compute_rho(Graph<T, S> *graph, thrust::device_vector<T> &delta_x,
              const T chi2, const T new_chi2, const T mu,
              const bool step_is_good) {
  // Compute rho
  //  TODO: Don't store these in the graph
  auto &b = graph->get_b();
  T num = (chi2 - new_chi2);
  T denom = mu * static_cast<T>(thrust::inner_product(
                     delta_x.begin(), delta_x.end(), delta_x.begin(),
                     static_cast<T>(0.0)));
  denom += static_cast<T>(thrust::inner_product(
      delta_x.begin(), delta_x.end(), b.begin(), static_cast<T>(0.0)));
  if (step_is_good) {
    denom += std::numeric_limits<T>::epsilon();
  } else {
    denom = 1;
  }
  return num / (denom);
}

template <typename T, typename S>
bool levenberg_marquardt(Graph<T, S> *graph, Solver<T, S> *solver,
                         const size_t num_iterations, T damping_factor,
                         uint8_t optimization_level, StreamPool &streams) {

  // Initialize something for all iterations
  auto start = std::chrono::steady_clock::now();
  T mu = damping_factor;
  T nu = 2;

  if (!graph->initialize_optimization(optimization_level)) {
    return false;
  }

  if (!graph->build_structure()) {
    return false;
  }

  graph->linearize(streams);

  thrust::device_vector<T> delta_x(graph->get_hessian_dimension());

  bool run = true;

  double time =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  // Print iteration table headers
  std::cout << std::setprecision(4) << std::setw(10) << "Iteration" << std::setw(16) << "Initial Chi2"
            << std::setw(16) << "Current Chi2" << std::setw(16) << "Lambda"
            << std::setw(16) << "Time" << std::setw(16) << "Total Time"
            << std::endl;
  std::cout << "---------------------------------------------------------------"
               "---------------------------"
            << std::endl;

  for (size_t i = 0; i < num_iterations && run; i++) {

    start = std::chrono::steady_clock::now();
    T chi2 = graph->chi2();

    if (!solver->solve(graph, delta_x.data().get(), static_cast<T>(mu),
                       streams)) {
      std::cerr << "Solver failed" << std::endl;
      return false;
    }

    // print delta x
    // thrust::host_vector<S> hx = delta_x;
    // std::cout << "Delta x: ";
    // for (size_t j = 0; j < hx.size(); j++) {
    //   std::cout << (T)hx[j] << " ";
    // }
    // std::cout << std::endl;

    graph->backup_parameters();
    graph->apply_step(delta_x.data().get(), streams);

    // Try step
    graph->compute_error();
    T new_chi2 = graph->chi2();

    bool step_is_good = std::isfinite(new_chi2);

    T rho = compute_rho(graph, delta_x, chi2, new_chi2, mu, step_is_good);

    if (step_is_good && rho > 0) {
      // update hyperparameters
      mu *= std::max(1.0 / 3.0, 1.0 - pow(2.0 * rho - 1, 3));
      nu = 2;
      // Relinearize since step is accepted
      graph->linearize(streams);
      // std::cout << "Good step" << std::endl;
      // std::cout << "rho: " << rho << std::endl;
    } else {
      graph->revert_parameters();
      graph->compute_error();
      // update hyperparameters
      mu *= nu;
      nu *= 2;
      // std::cout << "Bad step" << std::endl;
      new_chi2 = chi2;
    }

    // mu = std::clamp(mu, static_cast<T>(1e-12), static_cast<T>(1e12));

    double iteration_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    time += iteration_time;
    std::cout << std::setprecision(4) << std::setw(10) << i << std::setw(16) << chi2 << std::setw(16)
              << new_chi2 << std::setw(16) << mu << std::setw(16)
              << iteration_time << std::setw(16) << time << std::endl;

    if (!std::isfinite(mu)) {
      std::cout << "Damping factor is infinite, terminating optimization"
                << std::endl;
      run = false;
    }
  }

  // Should only really do this when optimization is successful
  graph->to_host();

  return run;
}

} // namespace optimizer
} // namespace glso