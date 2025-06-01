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

template <typename T, typename S> class Solver {
public:
  virtual ~Solver() = default;

  virtual bool solve(Graph<T, S> *graph, T *delta_x, T damping_factor) = 0;
};

template <typename T, typename S> class PCGSolver : public Solver<T, S> {
private:
  thrust::device_vector<T> v;

  // Need vectors for residuals of each factor
  thrust::device_vector<T> v1;   // v1 = Jv (dimension same as r)
  thrust::device_vector<T> v2;   // v2 = J^T v1 (dimension same as x)
  thrust::device_vector<T> r;    // residual
  thrust::device_vector<T> p;    // search direction
  thrust::device_vector<T> z;    // preconditioned residual
  thrust::device_vector<T> diag; // diagonal of Hessian
  thrust::device_vector<T> x_backup;
  thrust::device_vector<T> y;

  size_t max_iter;
  T tol;
  T rejection_ratio;

  Preconditioner<T, S> *preconditioner;

public:
  PCGSolver(size_t max_iter, T tol, T rejection_ratio,
            Preconditioner<T, S> *preconditioner)
      : max_iter(max_iter), tol(tol), rejection_ratio(rejection_ratio),
        preconditioner(preconditioner) {}

  // Assumes that x is already initialized
  virtual bool solve(Graph<T, S> *graph, T *x, T damping_factor) override {

    auto &vertex_descriptors = graph->get_vertex_descriptors();
    auto &factor_descriptors = graph->get_factor_descriptors();
    auto visitor = GraphVisitor<T, S>();
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
    T min_diag = static_cast<T>(1.0e-6);
    T max_diag = static_cast<T>(1.0e32);
    clamp(dim_h, min_diag, max_diag, diag.data().get());

    damp_by_factor(dim_h, v2.data().get(), damping_factor, diag.data().get(),
                   x);

    // 4. Finally r = b - v2
    axpy(dim_h, r.data().get(), (T)-1.0, (const T *)v2.data().get(), b);

    cudaDeviceSynchronize();

    // Rescale r
    y.resize(dim_h);
    auto rnorm = thrust::inner_product(thrust::device, r.begin(), r.end(), r.begin(),
                                      static_cast<T>(0.0));
    // if (rnorm > 1e-16) {
    rnorm = std::sqrt(rnorm);
    auto scale = 1.0 / rnorm;
    // scale = std::clamp(scale, 1.0e0, 1.0e0);
    rescale_vec<T>(dim_h, y.data().get(), scale, r.data().get());
    // Apply preconditioner
    preconditioner->precompute(visitor, vertex_descriptors, factor_descriptors,
                               dim_h, damping_factor);
    z.resize(dim_h);

    thrust::fill(z.begin(), z.end(), 0);
    preconditioner->apply(visitor, z.data().get(), y.data().get());

    p.resize(dim_h);
    thrust::copy(z.begin(), z.end(), p.begin()); // p = z

    x_backup.resize(dim_h);

    // 1. First compute dot(r, z)
    T rz = (T)thrust::inner_product(r.begin(), r.end(), z.begin(),
                                 static_cast<T>(0.0));
    // T rz_0 = rz;
    // T rz_0;

    // if constexpr (std::is_same<S, ghalf>::value) {
    //   rz_0 = CUDART_INF_FP16;
    // } else {
    //   rz_0 = std::numeric_limits<S>::infinity();
    // }

    T rz_0 = std::numeric_limits<T>::infinity();
    // T rz_0 = rz;

    /*
    // print r
    // std::cout << "Initial residual r: ";
    std::vector<T> r_host(dim_h);
    thrust::copy(r.begin(), r.end(), r_host.begin());
    // std::cout << "Initial residual r (first 10): ";
    // for (size_t i = 0; i < std::min(dim_h, size_t(10)); ++i) {
    //   std::cout << (T)r_host[i] << " ";
    // }
    // std::cout << std::endl;

    std::vector<T> z_host(dim_h);
    thrust::copy(z.begin(), z.end(), z_host.begin());
    // std::cout << "Initial preconditioned residual z (first 10): ";
    // for (size_t i = 0; i < std::min(dim_h, size_t(10)); ++i) {
    //   std::cout << (T)z_host[i] << " ";
    // }
    // std::cout << std::endl;

    std::vector<T> p_host(dim_h);
    thrust::copy(p.begin(), p.end(), p_host.begin());
    // std::cout << "Initial search direction p (first 10): ";
    // for (size_t i = 0; i < std::min(dim_h, size_t(10)); ++i) {
    //   std::cout << (T)p_host[i] << " ";
    // }
    // std::cout << std::endl;

    // check all three for NaN or Inf by iterating
    for (size_t i = 0; i < dim_h; ++i) {
      if (std::isnan((T)r_host[i]) || std::isinf((T)r_host[i])) {
      std::cerr << "NaN or Inf detected in initial residual r at index " << i
            << ": value = " << (T)r_host[i] << std::endl;
      return false;
      }
      if (std::isnan((T)z_host[i]) || std::isinf((T)z_host[i])) {
      std::cerr << "NaN or Inf detected in initial preconditioned residual z at index " << i
            << ": value = " << (T)z_host[i] << std::endl;
      return false;
      }
      if (std::isnan((T)p_host[i]) || std::isinf((T)p_host[i])) {
      std::cerr << "NaN or Inf detected in initial search direction p at index " << i
            << ": value = " << (T)p_host[i] << std::endl;
      return false;
      }
    }
    */


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
      T alpha = (rz) / thrust::inner_product(p.begin(), p.end(), v2.begin(),
                                           static_cast<T>(0.0));
      // 5. x  += alpha * p
      thrust::copy(thrust::device, x, x + dim_h, x_backup.begin());
      axpy(dim_h, x, alpha, p.data().get(), x);

      // 6. r -= alpha * v2
      axpy(dim_h, r.data().get(), -alpha, v2.data().get(),
           r.data().get());
      cudaDeviceSynchronize();

      rnorm = (T)thrust::inner_product(thrust::device, r.begin(), r.end(), r.begin(),
                                        static_cast<T>(0.0));
      // if (rnorm > 1e-16) {
      rnorm = std::sqrt(rnorm);
      scale = 1.0 / rnorm;
      // scale = std::clamp(scale, 1e0, 1.0e0);
      rescale_vec<T>(dim_h, y.data().get(), scale, r.data().get());
      // }


      // Apply preconditioner again
      thrust::fill(z.begin(), z.end(), 0);
      preconditioner->apply(visitor, z.data().get(), y.data().get());
      T rz_new = thrust::inner_product(r.begin(), r.end(), z.begin(),
                                       static_cast<T>(0.0));

      /*
      std::cout << "printing first 10 values of r and z after iteration " << k << std::endl;
      // print r
      thrust::copy(r.begin(), r.end(), r_host.begin());
      std::cout << "Residual r (first 10): ";
      for (size_t i = 0; i < std::min(dim_h, size_t(10)); ++i) {
        std::cout << (T)r_host[i] << " ";
      }
      std::cout << std::endl;
      // print z
      thrust::copy(z.begin(), z.end(), z_host.begin());
      std::cout << "Preconditioned residual z (first 10): ";
      for (size_t i = 0; i < std::min(dim_h, size_t(10)); ++i) {
        std::cout << (T)z_host[i] << " ";
      }
      std::cout << std::endl;

      thrust::copy(v1.begin(), v1.begin() + 10, r_host.begin());
      std::cout << "v1 (Jp) (first 10): ";
      for (size_t i = 0; i < std::min(dim_r, size_t(10)); ++i) {
        std::cout << (T)r_host[i] << " ";
      }
      std::cout << std::endl;

      thrust::copy(v2.begin(), v2.end(), r_host.begin());
      std::cout << "v2 (J^T v1) (first 10): ";
      for (size_t i = 0; i < std::min(dim_h, size_t(10)); ++i) {
        std::cout << (T)r_host[i] << " ";
      }
      std::cout << std::endl;
      
      std::cout << "alpha: " << static_cast<T>(alpha) << std::endl;



      std::cout << "Iteration " << k
                << ", rz_new: " << static_cast<T>(rz_new)
                << ", rz: " << static_cast<T>(rz)
                << ", rz_0: " << static_cast<T>(rz_0) << std::endl;

                */

      // 7. Check termination criteria
      // if (sqrt(rz_new / rz_0) < tol) {
      //   // std::cout << "Converged at iteration " << k << std::endl;
      //   break;
      // }

      // if (rz_new > rejection_ratio * rz_0) {
      if (std::abs(rz_new) > rejection_ratio * rz_0) {
        thrust::copy(thrust::device, x_backup.begin(), x_backup.end(), x);
        // std::cout << "Diverged at iteration " << k
        //           << ", resetting to previous x." << std::endl;
        //           std::cout << "rz_new: " << static_cast<T>(rz_new) << ", rz_0: " << static_cast<T>(rz_0) << ", rejection_ratio: " <<
        //           rejection_ratio << std::endl;
        break;
      }
      // rz_0 = std::min(rz_0, rz_new);
      rz_0 = std::min(rz_0, std::abs(rz_new));

      // 8. Compute beta
      T beta = rz_new / rz;
      // std::cout << "Iteration " << k << ", rz_new: " << static_cast<T>(rz_new)
      //           << ", rz_0: " << static_cast<T>(rz_0) << ", beta: "
      //           << static_cast<T>(beta) << std::endl;
      rz = rz_new;

      // 9. Update p
      axpy(dim_h, p.data().get(), beta, p.data().get(),
           z.data().get());
      cudaDeviceSynchronize();

      if (std::abs(static_cast<T>(rz_new)) < tol) {
        // std::cout << "Converged at iteration " << k << std::endl;
        // std::cout << "rz_new: " << static_cast<T>(rz_new) << ", tol: " << tol
        //           << std::endl;
        break;
      }
    }

    // TODO: Figure out failure cases
    return true;
  }
};

} // namespace glso