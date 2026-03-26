/// @file eigen_solver_interface.hpp
#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cstdint>
#include <type_traits>

namespace graphite {

// We need to make sure the eigen solver function calls are not
// compiled in a .cu file. So we declare specializations here and
// implement them in a .cpp file.
// TODO: Is there a better way to do this?

template <typename T>
using VecMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <typename Solver> Solver *create_eigen_ldlt_solver() {
  return nullptr;
}

template <typename Solver> void destroy_eigen_ldlt_solver(Solver *solver) {}

template <typename T, typename Index> class EigenLDLTSolverImpl;

template <>
EigenLDLTSolverImpl<double, int32_t> *
create_eigen_ldlt_solver<EigenLDLTSolverImpl<double, int32_t>>();

template <>
EigenLDLTSolverImpl<float, int32_t> *
create_eigen_ldlt_solver<EigenLDLTSolverImpl<float, int32_t>>();

template <>
EigenLDLTSolverImpl<double, int64_t> *
create_eigen_ldlt_solver<EigenLDLTSolverImpl<double, int64_t>>();

template <>
EigenLDLTSolverImpl<float, int64_t> *
create_eigen_ldlt_solver<EigenLDLTSolverImpl<float, int64_t>>();

template <>
void destroy_eigen_ldlt_solver<EigenLDLTSolverImpl<double, int32_t>>(
    EigenLDLTSolverImpl<double, int32_t> *solver);

template <>
void destroy_eigen_ldlt_solver<EigenLDLTSolverImpl<float, int32_t>>(
    EigenLDLTSolverImpl<float, int32_t> *solver);

template <>
void destroy_eigen_ldlt_solver<EigenLDLTSolverImpl<double, int64_t>>(
    EigenLDLTSolverImpl<double, int64_t> *solver);

template <>
void destroy_eigen_ldlt_solver<EigenLDLTSolverImpl<float, int64_t>>(
    EigenLDLTSolverImpl<float, int64_t> *solver);

template <typename T,
          typename Index =
              typename Eigen::SparseMatrix<T, Eigen::ColMajor>::StorageIndex>
class EigenLDLTWrapper {
private:
  static_assert(std::is_same<Index, int32_t>::value ||
                    std::is_same<Index, int64_t>::value,
                "EigenLDLTWrapper index type must be int32_t or int64_t");

  // using solver_impl = std::conditional_t<
  //     std::is_same<T, double>::value, EigenLDLTSolverImpl<double>,

  using solver_impl = EigenLDLTSolverImpl<T, Index>;

  solver_impl *solver;

public:
  EigenLDLTWrapper() { solver = create_eigen_ldlt_solver<solver_impl>(); }

  ~EigenLDLTWrapper() {
    destroy_eigen_ldlt_solver<solver_impl>(solver);
    solver = nullptr;
  }

  bool
  analyze_pattern(const Eigen::SparseMatrix<T, Eigen::ColMajor, Index> &matrix);
  bool factorize(const Eigen::SparseMatrix<T, Eigen::ColMajor, Index> &matrix);
  bool solve(const VecMap<T> &b, VecMap<T> &x);
};

} // namespace graphite