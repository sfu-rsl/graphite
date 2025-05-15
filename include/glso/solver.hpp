#pragma once
#include <glso/common.hpp>
#include <glso/factor.hpp>
#include <glso/kernel.hpp>
#include <thrust/inner_product.h>
#include <glso/utils.hpp>
#include <glso/preconditioner.hpp>

namespace glso {

    template<typename T>
    class PCGSolver {
        private:

        thrust::device_vector<T> v;
        
        // Need vectors for residuals of each factor
        // std::vector<thrust::device_vector<T>> v1; // Jv
        thrust::device_vector<T> v1; // v1 = Jv (dimension same as r)
        thrust::device_vector<T> v2; // v2 = J^T v1 (dimension same as x)
        thrust::device_vector<T> r; // residual
        thrust::device_vector<T> p; // search direction
        thrust::device_vector<T> diag; // diagonal of Hessian

        // thrust::device_vector<T> xout;

        public:

        // Assumes that x is already initialized
        void solve(GraphVisitor<T>& visitor, 
            std::vector<BaseVertexDescriptor<T>*>& vertex_descriptors,
            std::vector<BaseFactorDescriptor<T>*>& factor_descriptors,
            T* b,
            T* x, size_t dim_h, T damping_factor, size_t max_iter, T tol) {

            // std::cout << "Printing b: " << std::endl;
            // print_device_vector(b, dim_h);
            
            size_t dim_r = 0;
            v1.resize(factor_descriptors.size());
            for (size_t i = 0; i < factor_descriptors.size(); i++) {
                dim_r += factor_descriptors[i]->get_residual_size();
            }

            // std::cout << "Dim h: " << dim_h << std::endl;
            // std::cout << "Dim r: " << dim_r << std::endl;

            v1.resize(dim_r);
            thrust::fill(v1.begin(), v1.end(), 0);

            v2.resize(dim_h);
            thrust::fill(v2.begin(), v2.end(), 0);

            // xout.resize(dim_h);
            // thrust::fill(xout.begin(), xout.end(), 0);
            
            // Compute residual
            r.resize(dim_h); // dim h because dim(r) = dim(Ax) = dim(b)
            thrust::fill(r.begin(), r.end(), 0);

            // 1. First compute Jx
            T* v1_ptr = v1.data().get();
            for (size_t i = 0; i < factor_descriptors.size(); i++) {
                factor_descriptors[i]->visit_Jv(visitor, v1_ptr, x);
                v1_ptr += factor_descriptors[i]->get_residual_size();
            }
            cudaDeviceSynchronize();
            // Print v1 for debugging
            // std::cout << "v1: ";
            // for (size_t i = 0; i < v1.size(); i++) {
            //     std::cout << v1[i] << " ";
            // }
            // std::cout << std::endl;

            // 2. Then compute v2 = J^T v1
            v1_ptr = v1.data().get();
            for (size_t i = 0; i < factor_descriptors.size(); i++) {
                factor_descriptors[i]->visit_Jtv(visitor, v2.data().get(), v1_ptr);
                v1_ptr += factor_descriptors[i]->get_residual_size();
            }
            cudaDeviceSynchronize();
            // Print v2 for debugging
            // std::cout << "v2: ";
            // for (size_t i = 0; i < v2.size(); i++) {
            //     std::cout << v2[i] << " ";
            // }
            // std::cout << std::endl;

            // 3. Add damping factor
            // v2 += damping_factor * x
            // axpy(dim_h, v2.data().get(), damping_factor, x, v2.data().get());

            // v2 += damping_factor*diag(H)*x
            diag.resize(dim_h);
            thrust::fill(diag.begin(), diag.end(), 0);
            for (size_t i = 0; i < factor_descriptors.size(); i++) {
                factor_descriptors[i]->visit_scalar_diagonal(visitor, diag.data().get());
            }
            cudaDeviceSynchronize();
            thrust::fill(diag.begin(), diag.end(), 1.0);

            damp_by_factor(dim_h, v2.data().get(), damping_factor, diag.data().get(), x, v2.data().get());



            // 4. Finally r = b - v2
            axpy(dim_h, r.data().get(), -1.0, (const T*)v2.data().get(), b);
            // std::cout << "r: ";
            // for (size_t i = 0; i < r.size(); i++) {
            //     std::cout << r[i] << " ";
            // }
            // std::cout << std::endl;
            
            // std::cout << "b: ";
            // for (size_t i = 0; i < dim_h; i++) {
            //     std::cout << b[i] << " ";
            // }
            // std::cout << std::endl;

            cudaDeviceSynchronize();
            // thrust::copy(r.begin(), r.end(), p.begin());

            // Apply preconditioner
            // IdentityPreconditioner<T> preconditioner;
            BlockJacobiPreconditioner<T> preconditioner;
            preconditioner.precompute(visitor, vertex_descriptors, factor_descriptors, dim_h, damping_factor);
            thrust::device_vector<T> z(dim_h);
            thrust::fill(z.begin(), z.end(), 0);
            // std::cout << "Applying preconditioner" << std::endl;
            preconditioner.apply(visitor, z.data().get(), r.data().get());

            // thrust::host_vector<T> h_z(z);
            // std::cout << "First few values of h_z: ";
            // for (size_t i = 0; i < std::min<size_t>(10, h_z.size()); ++i) {
            //     std::cout << h_z[i] << " ";
            // }
            // std::cout << std::endl;
            
            p = z;
            
            // 1. First compute dot(r, z)
            T rz = thrust::inner_product(r.begin(), r.end(), z.begin(), 0.0);
            const T rz_0 = rz;

            // std::cout << "Starting PCG iterations" << std::endl;

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
                // axpy(dim_h, v2.data().get(), damping_factor, p.data().get(), v2.data().get());
                // v2 += damping_factor*diag(H)*p
                damp_by_factor(dim_h, v2.data().get(), damping_factor, diag.data().get(), p.data().get(), v2.data().get());
                // cudaDeviceSynchronize();

                // 4. Compute alpha = dot(r, z) / dot(p, v2)
                T alpha = rz / thrust::inner_product(p.begin(), p.end(), v2.begin(), 0.0);
                // std::cout << "alpha: " << alpha << ", rr: " << rr << ", dot(p, v2): " << thrust::inner_product(p.begin(), p.end(), v2.begin(), 0.0) << std::endl;
                // 5. x  += alpha * p
                axpy(dim_h, x, alpha, (const T*)p.data().get(), x);

                // 6. r -= alpha * v2
                axpy(dim_h, r.data().get(), -alpha, (const T*)v2.data().get(), r.data().get());
                cudaDeviceSynchronize();

                // // 7. Check termination criteria
                // T rr_new = thrust::inner_product(r.begin(), r.end(), r.begin(), 0.0);
                // if (sqrt(rr_new) < tol) {
                //     // std::cout << "Converged at iteration " << k << std::endl;
                //     break;
                // }


                // Apply preconditioner again
                thrust::fill(z.begin(), z.end(), 0);
                preconditioner.apply(visitor, z.data().get(), r.data().get());
                T rz_new = thrust::inner_product(r.begin(), r.end(), z.begin(), 0.0);

                // 7. Check termination criteria
                if (sqrt(rz_new / rz_0) < tol) {
                    // std::cout << "Converged at iteration " << k << std::endl;
                    break;
                }

                // 8. Compute beta
                T beta = rz_new / rz;
                rz = rz_new;

                // 9. Update p
                axpy(dim_h, p.data().get(), beta, (const T*)p.data().get(), r.data().get());
                cudaDeviceSynchronize();
            }
                
        
        
        }   



    };

}