#pragma once
#include <glso/common.hpp>
#include <glso/factor.hpp>
#include <glso/kernel.hpp>
#include <thrust/inner_product.h>

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

        thrust::device_vector<T> xout;

        public:

        void solve(GraphVisitor<T>& visitor, 
            std::vector<BaseVertexDescriptor<T>*>& vertex_descriptors,
            std::vector<BaseFactorDescriptor<T>*>& factor_descriptors,
            T* b,
            T* x0, size_t dim_h, size_t max_iter, T tol) {
            
            size_t dim_r = 0;
            v1.resize(factor_descriptors.size());
            for (size_t i = 0; i < factor_descriptors.size(); i++) {
                dim_r += factor_descriptors[i]->get_residual_size();
            }

            std::cout << "Dim h: " << dim_h << std::endl;
            std::cout << "Dim r: " << dim_r << std::endl;

            v1.resize(dim_r);
            thrust::fill(v1.begin(), v1.end(), 0);

            v2.resize(dim_h);
            thrust::fill(v2.begin(), v2.end(), 0);

            xout.resize(dim_h);
            thrust::fill(xout.begin(), xout.end(), 0);
            
            // Compute residual
            r.resize(dim_h); // dim h because dim(r) = dim(Ax) = dim(b)
            thrust::fill(r.begin(), r.end(), 0);

            // 1. First compute Jx
            T* v1_ptr = v1.data().get();
            for (size_t i = 0; i < factor_descriptors.size(); i++) {
                factor_descriptors[i]->visit_Jv(visitor, v1_ptr, x0);
                v1_ptr += factor_descriptors[i]->get_residual_size();
            }
            // cudaDeviceSynchronize();

            // 2. Then compute v2 = J^T v1
            v1_ptr = v1.data().get();
            for (size_t i = 0; i < factor_descriptors.size(); i++) {
                factor_descriptors[i]->visit_Jtv(visitor, v2.data().get(), v1_ptr);
                v1_ptr += factor_descriptors[i]->get_residual_size();
            }
            cudaDeviceSynchronize();

            // 3. Finally r = b - v2
            saxpy(dim_h, r.data().get(), -1.0, (const T*)v2.data().get(), b);
            cudaDeviceSynchronize();
            // thrust::copy(r.begin(), r.end(), p.begin());
            p = r;

            for (size_t k = 0; k < max_iter; k++) {


                // 1. First compute dot(r, r)
                T rr = thrust::inner_product(r.begin(), r.end(), r.begin(), 0.0);

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
                // 4. Compute alpha = dot(r, r) / dot(p, v2)
                T alpha = rr / thrust::inner_product(p.begin(), p.end(), v2.begin(), 0.0);

                // 5. x  += alpha * p
                saxpy(dim_h, xout.data().get(), alpha, (const T*)p.data().get(), xout.data().get());

                // 6. r -= alpha * v2
                saxpy(dim_h, r.data().get(), -alpha, (const T*)v2.data().get(), r.data().get());
                cudaDeviceSynchronize();

                // 7. Check termination criteria
                T rr_new = thrust::inner_product(r.begin(), r.end(), r.begin(), 0.0);
                if (sqrt(rr_new) < tol) {
                    std::cout << "Converged at iteration " << k << std::endl;
                    break;
                }
                // 8. Compute beta
                T beta = rr_new / rr;

                // 9. Update p
                saxpy(dim_h, p.data().get(), beta, (const T*)p.data().get(), r.data().get());
                cudaDeviceSynchronize();
            }
                
        
        
        }   



    };

}