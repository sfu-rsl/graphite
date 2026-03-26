/// @file eigen_solver.cpp
#include <graphite/solver/eigen_solver_interface.hpp>

namespace graphite {

template <class Method>
class Decomp : public Method {
 public:
};


template <typename T, typename Index>
class EigenLDLTSolverImpl {
  public:

  using decomp_method =
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<T, Eigen::ColMajor, Index>,
                            Eigen::Upper>;
  Decomp<decomp_method> decomp;

  bool analyze_pattern(
      const Eigen::SparseMatrix<T, Eigen::ColMajor, Index> &matrix) {
    decomp.analyzePattern(matrix);
    return decomp.info() == Eigen::Success;
  }

  bool factorize(const Eigen::SparseMatrix<T, Eigen::ColMajor, Index> &matrix) {
    decomp.factorize(matrix);
    return decomp.info() == Eigen::Success;
  }

  bool solve(const VecMap<T> &b,
             VecMap<T> &x) {
        x = decomp.solve(b);
        return decomp.info() == Eigen::Success;            
    }
};

// Implementation for solver creation

template <>
EigenLDLTSolverImpl<double, int32_t> *
create_eigen_ldlt_solver<EigenLDLTSolverImpl<double, int32_t>>() {
  return new EigenLDLTSolverImpl<double, int32_t>();
}

template <>
EigenLDLTSolverImpl<float, int32_t> *
create_eigen_ldlt_solver<EigenLDLTSolverImpl<float, int32_t>>() {
  return new EigenLDLTSolverImpl<float, int32_t>();
}

template <>
EigenLDLTSolverImpl<double, int64_t> *
create_eigen_ldlt_solver<EigenLDLTSolverImpl<double, int64_t>>() {
  return new EigenLDLTSolverImpl<double, int64_t>();
}

template <>
EigenLDLTSolverImpl<float, int64_t> *
create_eigen_ldlt_solver<EigenLDLTSolverImpl<float, int64_t>>() {
  return new EigenLDLTSolverImpl<float, int64_t>();
}

// Implementation for solver destruction
template <>
void destroy_eigen_ldlt_solver<EigenLDLTSolverImpl<double, int32_t>>(
    EigenLDLTSolverImpl<double, int32_t> *solver) {
    delete solver;
}

template <>
void destroy_eigen_ldlt_solver<EigenLDLTSolverImpl<float, int32_t>>(
    EigenLDLTSolverImpl<float, int32_t> *solver) {
    delete solver;
}

template <>
void destroy_eigen_ldlt_solver<EigenLDLTSolverImpl<double, int64_t>>(
    EigenLDLTSolverImpl<double, int64_t> *solver) {
    delete solver;
}

template <>
void destroy_eigen_ldlt_solver<EigenLDLTSolverImpl<float, int64_t>>(
    EigenLDLTSolverImpl<float, int64_t> *solver) {
    delete solver;
}

template <typename T, typename Index>
bool EigenLDLTWrapper<T, Index>::analyze_pattern(
    const Eigen::SparseMatrix<T, Eigen::ColMajor, Index> &matrix) {
  return solver->analyze_pattern(matrix);
}

template <typename T, typename Index>
bool EigenLDLTWrapper<T, Index>::factorize(
    const Eigen::SparseMatrix<T, Eigen::ColMajor, Index> &matrix) {
  return solver->factorize(matrix);
}

template <typename T, typename Index>
bool EigenLDLTWrapper<T, Index>::solve(const VecMap<T> &b, VecMap<T> &x) {
  return solver->solve(b, x);
}

template class EigenLDLTWrapper<double, int32_t>;
template class EigenLDLTWrapper<float, int32_t>;
template class EigenLDLTWrapper<double, int64_t>;
template class EigenLDLTWrapper<float, int64_t>;

} // namespace graphite