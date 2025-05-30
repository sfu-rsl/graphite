#pragma once
#include <glso/common.hpp>
#include <glso/factor.hpp>
#include <glso/kernel.hpp>
#include <glso/preconditioner.hpp>
#include <glso/utils.hpp>
#include <memory>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>

namespace glso {

template <typename T> class Solver {
public:
  virtual ~Solver() = default;

  virtual bool solve(Graph<T> *graph, T *delta_x, T damping_factor) = 0;
};

template <typename T> class PCGSolver : public Solver<T> {
private:
  thrust::device_vector<T> v;

  // Need vectors for residuals of each factor
  thrust::device_vector<T> v1;   // v1 = Jv (dimension same as r)
  thrust::device_vector<T> v2;   // v2 = J^T v1 (dimension same as x)
  thrust::device_vector<T> r;    // residual
  thrust::device_vector<T> p;    // search direction
  thrust::device_vector<T> diag; // diagonal of Hessian
  thrust::device_vector<T> x_backup;

  size_t max_iter;
  T tol;
  T rejection_ratio;

  Preconditioner<T> *preconditioner;

public:
  PCGSolver(size_t max_iter, T tol, T rejection_ratio, Preconditioner<T> *preconditioner)
      : max_iter(max_iter), tol(tol), rejection_ratio(rejection_ratio), preconditioner(preconditioner) {}

  // Assumes that x is already initialized
  virtual bool solve(Graph<T> *graph, T *x, T damping_factor) override {

    auto &vertex_descriptors = graph->get_vertex_descriptors();
    auto &factor_descriptors = graph->get_factor_descriptors();
    auto visitor = GraphVisitor<T>();
    T *b = graph->get_b().data().get();
    size_t dim_h = graph->get_hessian_dimension();

    thrust::fill(thrust::device, x, x + dim_h, 0);

    size_t dim_r = 0;
    v1.resize(factor_descriptors.size());
    for (size_t i = 0; i < factor_descriptors.size(); i++) {
      dim_r += factor_descriptors[i]->get_residual_size();
    }

    v1.resize(dim_r);
    thrust::fill(v1.begin(), v1.end(), 0);

    v2.resize(dim_h);
    thrust::fill(v2.begin(), v2.end(), 0);

    // Compute residual
    r.resize(dim_h); // dim h because dim(r) = dim(Ax) = dim(b)
    thrust::fill(r.begin(), r.end(), 0);

    // 1. First compute Jx
    T *v1_ptr = v1.data().get();
    for (size_t i = 0; i < factor_descriptors.size(); i++) {
      factor_descriptors[i]->visit_Jv(visitor, v1_ptr, x);
      v1_ptr += factor_descriptors[i]->get_residual_size();
    }
    cudaDeviceSynchronize();

    // 2. Then compute v2 = J^T v1
    v1_ptr = v1.data().get();
    for (size_t i = 0; i < factor_descriptors.size(); i++) {
      factor_descriptors[i]->visit_Jtv(visitor, v2.data().get(), v1_ptr);
      v1_ptr += factor_descriptors[i]->get_residual_size();
    }
    cudaDeviceSynchronize();

    // 3. Add damping factor
    // v2 += damping_factor*diag(H)*x
    diag.resize(dim_h);
    thrust::fill(diag.begin(), diag.end(), 0);
    for (size_t i = 0; i < factor_descriptors.size(); i++) {
      factor_descriptors[i]->visit_scalar_diagonal(visitor, diag.data().get());
    }
    cudaDeviceSynchronize();
    // thrust::fill(diag.begin(), diag.end(), 1.0);

    // Check for negative values in diag and print an error if found
    T min_diag = 1.0e-6;
    T max_diag = 1.0e32;
    clamp(dim_h, min_diag, max_diag, diag.data().get());

    damp_by_factor(dim_h, v2.data().get(), damping_factor, diag.data().get(),
                   x);

    // 4. Finally r = b - v2
    axpy(dim_h, r.data().get(), (T)-1.0, (const T *)v2.data().get(), b);

    cudaDeviceSynchronize();

    // Apply preconditioner
    preconditioner->precompute(visitor, vertex_descriptors, factor_descriptors,
                               dim_h, damping_factor);
    thrust::device_vector<T> z(dim_h);

    thrust::fill(z.begin(), z.end(), 0);
    preconditioner->apply(visitor, z.data().get(), r.data().get());

    p.resize(dim_h);
    thrust::copy(z.begin(), z.end(), p.begin()); // p = z

    x_backup.resize(dim_h);

    // 1. First compute dot(r, z)
    T rz = thrust::inner_product(r.begin(), r.end(), z.begin(), 0.0);
    // T rz_min = rz;
    T rz_min = std::numeric_limits<T>::infinity();
    const T rz_0 = rz;
    const T relative_thresh = std::abs(rz_0 * tol);
    // constexpr T rejection_ratio = 5.0;
    for (size_t k = 0; k < max_iter; k++) {

      // 2. Compute v1 = Jp
      thrust::fill(v1.begin(), v1.end(), 0);
      v1_ptr = v1.data().get(); // reset
      for (size_t i = 0; i < factor_descriptors.size(); i++) {
        factor_descriptors[i]->visit_Jv(visitor, v1_ptr, p.data().get());
        v1_ptr += factor_descriptors[i]->get_residual_size();
      }
      // cudaDeviceSynchronize();

      // 3. Compute v2 = J^T v1
      thrust::fill(v2.begin(), v2.end(), 0);
      v1_ptr = v1.data().get(); // reset
      for (size_t i = 0; i < factor_descriptors.size(); i++) {
        factor_descriptors[i]->visit_Jtv(visitor, v2.data().get(), v1_ptr);
        v1_ptr += factor_descriptors[i]->get_residual_size();
      }
      cudaDeviceSynchronize();
      // Add damping factor
      // v2 += damping_factor*diag(H)*p
      damp_by_factor(dim_h, v2.data().get(), damping_factor, diag.data().get(),
                     p.data().get());
      // cudaDeviceSynchronize();

      // 4. Compute alpha = dot(r, z) / dot(p, v2)
      T alpha = rz / thrust::inner_product(p.begin(), p.end(), v2.begin(), 0.0);
      // 5. x  += alpha * p
      thrust::copy(thrust::device, x, x + dim_h, x_backup.begin());
      axpy(dim_h, x, alpha, (const T *)p.data().get(), x);

      // 6. r -= alpha * v2
      axpy(dim_h, r.data().get(), -alpha, (const T *)v2.data().get(),
           r.data().get());
      cudaDeviceSynchronize();

      // Apply preconditioner again
      thrust::fill(z.begin(), z.end(), 0);
      preconditioner->apply(visitor, z.data().get(), r.data().get());
      T rz_new = thrust::inner_product(r.begin(), r.end(), z.begin(), 0.0);

      // 7. Check termination criteria
      // if (sqrt(rz_new / rz_min) < tol) {
      //   // std::cout << "Converged at iteration " << k << std::endl;
      //   break;
      // }

      if (std::abs(rz_new) >= rejection_ratio * rz_min) {
        thrust::copy(thrust::device, x_backup.begin(), x_backup.end(), x);
        break;
      }
      rz_min = std::min(rz_min, std::abs(rz_new));

      // 8. Compute beta
      T beta = rz_new / rz;
      rz = rz_new;

      // 9. Update p
      axpy(dim_h, p.data().get(), beta, (const T *)p.data().get(),
           z.data().get());
      cudaDeviceSynchronize();

      // std::cout << "Termination criteria at iteration " << k
      //           << ": rz_new = " << rz_new << ", tol = " << tol << std::endl;
      //           std::cout << "rz_0 = " << rz_0 << std::endl;
      // if (std::abs(rz_new) < tol) {
      //   break;
      // }
      if (std::abs(rz_new) <= relative_thresh) {
        // std::cout << "Converged at iteration " << k << std::endl;
        break;
      }
    }

    // TODO: Figure out failure cases
    return true;
  }
};

} // namespace glso