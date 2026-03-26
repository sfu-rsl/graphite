#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/src/Core/Matrix.h>

namespace graphite::test_helpers {

using SparseMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;

struct SchurReference {
  SparseMat hpl;
  SparseMat hplt;
  SparseMat hll_inv;
  SparseMat schur_upper;
};

SchurReference build_schur_reference(const SparseMat &hessian_sym,
                                     size_t pose_start, size_t pose_dim,
                                     size_t landmark_start,
                                     size_t landmark_dim);

Eigen::VectorXd compute_b_schur_cpu(const Eigen::VectorXd &b, const SparseMat &hpl,
                         const SparseMat &hll_inv, size_t pose_dim,
                         size_t landmark_start, size_t landmark_dim);

Eigen::VectorXd compute_landmark_update_cpu(const Eigen::VectorXd &b, const SparseMat &hplt,
                                            const SparseMat &hll_inv,
                                 const Eigen::VectorXd &dx_p,
                                 size_t landmark_start,
                                 size_t landmark_dim);

} // namespace graphite::test_helpers
