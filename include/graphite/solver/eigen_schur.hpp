/// @file eigen_schur_ldlt.hpp
#pragma once

#include <graphite/hessian.hpp>
#include <graphite/schur.hpp>
#include <graphite/solver/eigen_solver_interface.hpp>
#include <graphite/solver/solver.hpp>

namespace graphite {

template <typename T, typename S,
          typename Index =
              typename Eigen::SparseMatrix<S, Eigen::ColMajor>::StorageIndex>
class EigenSchurLDLTSolver : public Solver<T, S> {
private:
  EigenLDLTWrapper<S, Index> solver;

  Eigen::SparseMatrix<S, Eigen::ColMajor, Index> matrix;

  Hessian<T, S> H;
  SchurComplement<T, S> schur;
  CSCMatrix<S, Index> d_matrix;

  thrust::host_vector<S> h_x;
  thrust::host_vector<S> h_b;

  void fill_matrix_structure() {
    const auto dim = d_matrix.d_pointers.size() - 1;
    matrix.resize(dim, dim);
    matrix.resizeNonZeros(d_matrix.d_values.size());

    auto h_ptrs = matrix.outerIndexPtr();
    auto h_indices = matrix.innerIndexPtr();

    thrust::copy(d_matrix.d_pointers.begin(), d_matrix.d_pointers.end(),
                 h_ptrs);
    thrust::copy(d_matrix.d_indices.begin(), d_matrix.d_indices.end(),
                 h_indices);

    h_x.resize(dim);
    h_b.resize(dim);
  }

  void fill_matrix_values() {
    auto h_values = matrix.valuePtr();
    thrust::copy(d_matrix.d_values.begin(), d_matrix.d_values.end(), h_values);
  }

public:
  EigenSchurLDLTSolver() : solver(), schur(H) {}

  virtual void update_structure(Graph<T, S> *graph,
                                StreamPool &streams) override {
    H.build_structure(graph, streams);
    schur.build_structure(graph, streams);
    schur.build_csc_structure(graph, d_matrix);
    fill_matrix_structure();
    solver.analyze_pattern(matrix);
  }

  virtual void update_values(Graph<T, S> *graph, StreamPool &streams) override {
    H.update_values(graph, streams);
  }

  virtual void set_damping_factor(Graph<T, S> *graph, T damping_factor,
                                  const bool use_identity,
                                  StreamPool &streams) override {
    H.apply_damping(graph, damping_factor, use_identity, streams);
  }

  virtual bool solve(Graph<T, S> *graph, T *x, StreamPool &streams) override {

    // Update matrix values here (to avoid extra work when damping
    schur.update_values(graph, streams);
    schur.update_csc_values(graph, d_matrix);
    fill_matrix_values();

    if (!solver.factorize(matrix)) {
      std::cerr << "Schur LDLT matrix decomposition failed!";
      return false;
    }

    const auto dim = graph->get_hessian_dimension();
    const auto &offsets = graph->get_offset_vector();
    const auto p_block_col = schur.lowest_eliminated_block_col;
    const auto p_dim = offsets[p_block_col];

    thrust::fill(thrust::device, x, x + dim, static_cast<T>(0.0));

    thrust::copy(schur.get_b_Schur().begin(), schur.get_b_Schur().end(),
                 h_b.begin());

    thrust::device_ptr<T> d_x(x);
    thrust::copy(d_x, d_x + p_dim, h_x.data());

    auto map_b = VecMap<S>(h_b.data(), p_dim, 1);
    auto map_x = VecMap<S>(h_x.data(), p_dim, 1);
    if (!solver.solve(map_b, map_x)) {
      std::cerr << "Schur LDLT solve failed!";
      return false;
    }

    thrust::copy(h_x.begin(), h_x.end(), d_x);

    schur.compute_landmark_update(graph, streams, x + p_dim, x);

    return true;
  }
};

} // namespace graphite
