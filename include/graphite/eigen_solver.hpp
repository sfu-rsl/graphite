#pragma once
#include <graphite/eigen_solver_interface.hpp>
#include <graphite/solver.hpp>

namespace graphite {

template <typename T, typename S>
class EigenLDLTSolver : public Solver<T, S> {
 private:
  EigenLDLTWrapper<S> solver;

  Eigen::SparseMatrix<S, Eigen::ColMajor> matrix;

  using Index = typename Eigen::SparseMatrix<S, Eigen::ColMajor>::StorageIndex;


  Hessian<T, S> H;
  CSCMatrix<S, Index> d_matrix;

  thrust::host_vector<S> h_x;
  thrust::host_vector<S> h_b;

  void fill_matrix_structure() {
    const auto dim = d_matrix.d_pointers.size() - 1;
    matrix.resize(dim, dim);
    matrix.resizeNonZeros(d_matrix.d_values.size());

    auto h_ptrs = matrix.outerIndexPtr();
    auto h_indices = matrix.innerIndexPtr();

    thrust::copy(thrust::device, d_matrix.d_pointers.begin(), d_matrix.d_pointers.end(), h_ptrs);
    thrust::copy(thrust::device, d_matrix.d_indices.begin(), d_matrix.d_indices.end(), h_indices);

    h_x.resize(dim);
    h_b.resize(dim);
  }

  void fill_matrix_values() {
    auto h_values = matrix.valuePtr();
    thrust::copy(thrust::device, d_matrix.d_values.begin(), d_matrix.d_values.end(), h_values);
  }

 public:
  EigenLDLTSolver(): solver() {}

  virtual void update_structure(Graph<T, S> *graph, StreamPool &streams) override {
    H.build_structure(graph, streams);
    H.build_csc_structure(graph, d_matrix);
    fill_matrix_structure(); // for CPU matrix
    solver.analyze_pattern(matrix);
  }

  virtual void update_values(Graph<T, S> *graph, StreamPool &streams) override {
    H.update_values(graph, streams);
    H.update_csc_values(graph, d_matrix);
    fill_matrix_values(); // for CPU matrix
  }

  virtual void set_damping_factor(Graph<T, S> *graph, T damping_factor, StreamPool &streams) override {
    H.apply_damping(graph, damping_factor, streams);
    H.update_csc_values(graph, d_matrix);
    fill_matrix_values(); // TODO: Use a more lightweight method to just update diagonal
  }

  virtual bool solve(Graph<T, S> *graph, T *x, StreamPool &streams) override {
    
    auto dim = graph->get_hessian_dimension();
    // std::cout << "Printing system matrix:" << std::endl;
    // std::cout << matrix << std::endl;

    if (!solver.factorize(matrix)) {
      std::cerr << "LDLT matrix decomposition failed!";
      return false;
    }

    // std::cout << "LDLT success!" << std::endl;
    // Not sure if zeroing is required
    thrust::fill(thrust::device, x, x + dim, static_cast<T>(0.0));

    // std::cout << "Copying b and x to host" << std::endl;
    thrust::copy(thrust::device, graph->get_b().begin(), graph->get_b().end(), h_b.data());
    thrust::copy(thrust::device, x, x + dim, h_x.data());

    auto map_b = VecMap<S>(h_b.data(), dim, 1);
    auto map_x = VecMap<S>(h_x.data(), dim, 1);
    // std::cout << "Solving!" << std::endl;
    solver.solve(map_b, map_x);
    // std::cout << "Copying back solution!" << std::endl;
    // Copy x back to device
    thrust::copy(thrust::device, h_x.begin(), h_x.end(), x);
    // std::cout << "Solution copied back!" << std::endl;

    return true;
  }
};

}  // namespace graphite