#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <graphite/solver.hpp>
// Interface for Linear Solvers
namespace graphite {

template <class Method>
class Decomp : public Method {
 public:
};

template <typename T, typename S>
class EigenLDLTSolver : public Solver<T, S> {
 private:
  using decomp_method =
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<S, Eigen::ColMajor>,
                            Eigen::Upper>;
  Decomp<decomp_method> decomp;

  Eigen::SparseMatrix<S, Eigen::ColMajor> matrix;

  using Index = typename Eigen::SparseMatrix<S, Eigen::ColMajor>::StorageIndex;

  // thrust::host_vector<Index> h_ptrs;
  // thrust::host_vector<size_t> h_indices;
  // thrust::host_vector<S> h_values;

  Hessian<T, S> H;
  CSRMatrix<S, Index> d_matrix;

  thrust::host_vector<S> h_x;
  thrust::host_vector<S> h_b;

  void fill_matrix_structure() {
    const auto dim = d_matrix.d_row_pointers.size() - 1;
    matrix.resize(dim, dim);
    matrix.resizeNonZeros(d_matrix.d_values.size());

    auto h_ptrs = matrix.outerIndexPtr();
    auto h_indices = matrix.innerIndexPtr();

    thrust::copy(thrust::device, d_matrix.d_row_pointers.begin(), d_matrix.d_row_pointers.end(), h_ptrs);
    thrust::copy(thrust::device, d_matrix.d_col_indices.begin(), d_matrix.d_col_indices.end(), h_indices);

    h_x.resize(dim);
    h_b.resize(dim);
  }

  void fill_matrix_values() {
    auto h_values = matrix.valuePtr();
    thrust::copy(thrust::device, d_matrix.d_values.begin(), d_matrix.d_values.end(), h_values);
  }

 public:
  EigenLDLTSolver() {}

  virtual void update_structure(Graph<T, S> *graph, StreamPool &streams) override {
    H.build_structure(graph, streams);
    H.build_csr_structure(graph, d_matrix);
    fill_matrix_structure(); // for CPU matrix
    decomp.analyzePattern(matrix);
  }

  virtual void update_values(Graph<T, S> *graph, StreamPool &streams) override {
    H.update_values(graph, streams);
    H.update_csr_values(graph, d_matrix);
    fill_matrix_values(); // for CPU matrix
  }

  virtual void set_damping_factor(Graph<T, S> *graph, T damping_factor, StreamPool &streams) override {
    H.apply_damping(graph, damping_factor, streams);
    H.update_csr_values(graph, d_matrix);
    fill_matrix_values(); // TODO: Use a more lightweight method to just update diagonal
  }

  virtual bool solve(Graph<T, S> *graph, T *x, StreamPool &streams) override {
    
    auto dim = graph->get_hessian_dimension();
    // std::cout << "Printing system matrix:" << std::endl;
    // std::cout << matrix << std::endl;

    decomp.factorize(matrix);
    if (decomp.info() != Eigen::Success) {
      std::cerr << "LDLT matrix decomposition failed!";
      return false;
    }

    // std::cout << "LDLT success!" << std::endl;
    // Not sure if zeroing is required
    thrust::fill(thrust::device, x, x + dim, static_cast<T>(0.0));

    // std::cout << "Copying b and x to host" << std::endl;
    thrust::copy(thrust::device, graph->get_b().begin(), graph->get_b().end(), h_b.data());
    thrust::copy(thrust::device, x, x + dim, h_x.data());

    auto map_b = Eigen::Map<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>>(h_b.data(), dim, 1);
    auto map_x = Eigen::Map<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>>(h_x.data(), dim, 1);
    // std::cout << "Solving!" << std::endl;
    map_x = decomp.solve(map_b);
    // std::cout << "Copying back solution!" << std::endl;
    // Copy x back to device
    thrust::copy(thrust::device, h_x.begin(), h_x.end(), x);
    // std::cout << "Solution copied back!" << std::endl;

    return true;
  }
};

}  // namespace graphite