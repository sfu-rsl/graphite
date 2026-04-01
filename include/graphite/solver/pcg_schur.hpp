/// @file pcg_schur.hpp
#pragma once

#include <cublas_v2.h>
#include <graphite/hessian.hpp>
#include <graphite/ops/schur.hpp>
#include <graphite/ops/vector.hpp>
#include <graphite/preconditioner/schur_preconditioner.hpp>
#include <graphite/schur.hpp>
#include <graphite/solver/solver.hpp>
#include <ratio>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace graphite {
/// @brief PCG solver using the Schur complement
template <typename T, typename S> class PCGSchurSolver : public Solver<T, S> {
private:
  Hessian<T, S> H;
  SchurComplement<T, S> schur;
  SchurPreconditioner<T, S> *preconditioner;

  thrust::device_vector<T> r;
  thrust::device_vector<T> p;
  thrust::device_vector<T> z;
  thrust::device_vector<T> Ap;
  thrust::device_vector<T> x_backup;

  size_t pose_dim;
  size_t max_iter;
  T tol;
  T rejection_ratio;

public:
  PCGSchurSolver(size_t max_iter, T tol, T rejection_ratio,
                 SchurPreconditioner<T, S> *preconditioner)
      : schur(H), pose_dim(0), max_iter(max_iter), tol(tol),
        rejection_ratio(rejection_ratio), preconditioner(preconditioner) {}

  ~PCGSchurSolver() = default;

  virtual void update_structure(Graph<T, S> *graph,
                                StreamPool &streams) override {
    H.build_structure(graph, streams);
    schur.build_structure(graph, streams);
    schur.setup_schur_vector_multiply(graph, streams);

    const auto &offsets = graph->get_offset_vector();
    pose_dim = offsets[schur.lowest_eliminated_block_col];

    preconditioner->update_structure(graph, &schur, streams);

    r.resize(pose_dim);
    p.resize(pose_dim);
    z.resize(pose_dim);
    Ap.resize(pose_dim);
    x_backup.resize(pose_dim);
  }

  virtual void update_values(Graph<T, S> *graph, StreamPool &streams) override {
    H.update_values(graph, streams);
  }

  virtual void set_damping_factor(Graph<T, S> *graph, T damping_factor,
                                  const bool use_identity,
                                  StreamPool &streams) override {
    H.apply_damping(graph, damping_factor, use_identity, streams);
    preconditioner->set_damping_factor(graph, &schur, damping_factor,
                                       use_identity, streams);
  }

  virtual bool solve(Graph<T, S> *graph, T *x, StreamPool &streams) override {
    // We need to defer all recomputation (update_values) until this point to
    // avoid excessive recomputation (e.g. when modifying the diagonal)
    schur.update_values(graph, streams);
    preconditioner->update_values(graph, &schur, streams);

    // Now initialize vectors for PCG loop

    const cudaStream_t stream = streams.select(0);
    const auto dim_h = graph->get_hessian_dimension();

    thrust::fill(thrust::cuda::par_nosync.on(stream), x, x + dim_h,
                 static_cast<T>(0));

    thrust::copy(thrust::cuda::par_nosync.on(stream),
                 schur.get_b_Schur().begin(), schur.get_b_Schur().end(),
                 r.begin());

    cudaStreamSynchronize(stream);

    preconditioner->apply(graph, &schur, z.data().get(), r.data().get(),
                          streams);
    thrust::copy(thrust::cuda::par_nosync.on(stream), z.begin(), z.end(),
                 p.begin());

    T rz = thrust::inner_product(thrust::cuda::par.on(stream), r.begin(),
                                 r.end(), z.begin(), static_cast<T>(0));
    T rz_0 = std::numeric_limits<T>::infinity();

    for (size_t k = 0; k < max_iter; ++k) {
      if (rz == static_cast<T>(0)) {
        break;
      }

      // 3. Compute Schur*p
      schur.execute_schur_vector_multiply(graph, streams, Ap.data().get(),
                                          p.data().get());

      const T denom =
          thrust::inner_product(thrust::cuda::par.on(stream), p.begin(),
                                p.end(), Ap.begin(), static_cast<T>(0));
      if (denom == static_cast<T>(0) || std::isnan(denom)) {
        break;
      }

      // 4. Compute alpha = dot(r, z) / dot(p, v2)
      const T alpha = rz / denom;

      // 5. x  += alpha * p
      thrust::copy(thrust::cuda::par_nosync.on(stream), x, x + pose_dim,
                   x_backup.begin());

      ops::axpy_async(stream, pose_dim, x, alpha, p.data().get(), x);
      // 6. r -= alpha * v2
      ops::axpy_async(stream, pose_dim, r.data().get(), -alpha, Ap.data().get(),
                      r.data().get());

      cudaStreamSynchronize(stream);
      // Apply preconditioner again
      preconditioner->apply(graph, &schur, z.data().get(), r.data().get(),
                            streams);

      const T rz_new =
          thrust::inner_product(thrust::cuda::par.on(stream), r.begin(),
                                r.end(), z.begin(), static_cast<T>(0));
      if (std::abs(rz_new) > rejection_ratio * rz_0 || std::isnan(rz_new)) {
        thrust::copy(thrust::cuda::par.on(stream), x_backup.begin(),
                     x_backup.end(), x);
        break;
      }
      rz_0 = std::min(rz_0, std::abs(rz_new));

      // 8. Compute beta
      const T beta = rz_new / rz;
      rz = rz_new;

      // 9. Update p
      ops::axpy_async(stream, pose_dim, p.data().get(), beta, p.data().get(),
                      z.data().get());
      cudaStreamSynchronize(stream);

      if (std::abs(static_cast<T>(rz_new)) < tol) {
        break;
      }
    }

    cudaStreamSynchronize(stream);
    schur.compute_landmark_update(graph, streams, x + pose_dim, x);
    return true;
  }
};

} // namespace graphite