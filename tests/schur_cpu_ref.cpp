#include "schur_cpu_ref.hpp"

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>

namespace graphite::test_helpers {

SchurReference build_schur_reference(const SparseMat &hessian_sym,
                                     size_t pose_start, size_t pose_dim,
                                     size_t landmark_start,
                                     size_t landmark_dim) {
  SchurReference out;

  const SparseMat hpp =
      hessian_sym.block(pose_start, pose_start, pose_dim, pose_dim);
  out.hpl =
      hessian_sym.block(pose_start, landmark_start, pose_dim, landmark_dim);
  out.hplt = out.hpl.transpose();

  out.hll_inv = hessian_sym.block(landmark_start, landmark_start, landmark_dim,
                                  landmark_dim);

  for (size_t i = 0; i < landmark_dim; i += 3) {
    const Eigen::Matrix3d block = out.hll_inv.block(i, i, 3, 3).toDense();
    const Eigen::Matrix3d block_inv = block.inverse();
    for (size_t r = 0; r < 3; ++r) {
      for (size_t c = 0; c < 3; ++c) {
        out.hll_inv.coeffRef(i + r, i + c) = block_inv(r, c);
      }
    }
  }

  const SparseMat schur = hpp - out.hpl * out.hll_inv * out.hplt;
  out.schur_upper = SparseMat(schur.template triangularView<Eigen::Upper>());
  return out;
}

Eigen::VectorXd compute_b_schur_cpu(const Eigen::VectorXd &b, const SparseMat &hpl,
                         const SparseMat &hll_inv, size_t pose_dim,
                         size_t landmark_start, size_t landmark_dim) {
  return b.head(pose_dim) -
         hpl * hll_inv * b.segment(landmark_start, landmark_dim);
}

Eigen::VectorXd compute_landmark_update_cpu(const Eigen::VectorXd &b, const SparseMat &hplt,
                                 const SparseMat &hll_inv,
                                 const Eigen::VectorXd &dx_p,
                                 size_t landmark_start,
                                 size_t landmark_dim) {
  return hll_inv * (b.segment(landmark_start, landmark_dim) - hplt * dx_p);
}

} // namespace graphite::test_helpers
