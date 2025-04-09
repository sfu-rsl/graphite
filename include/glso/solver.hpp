#pragma once
#include <glso/common.hpp>
#include <glso/factor.hpp>
#include <glso/kernel.hpp>

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

        thrust::device_vector<T> xout;

        public:

        void solve(GraphVisitor<T>& visitor, 
            std::vector<BaseVertexDescriptor<T>*>& vertex_descriptors,
            std::vector<BaseFactorDescriptor<T>*>& factor_descriptors,
            T* b,
            T* x0, size_t dim_h, size_t max_iter) {
            
            size_t dim_r = 0;
            v1.resize(factor_descriptors.size());
            for (size_t i = 0; i < factor_descriptors.size(); i++) {
                // v1[i].resize(factor_descriptors[i]->get_residual_size());
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
            cudaDeviceSynchronize();

            // 2. Then compute v2 = J^T v1
            v1_ptr = v1.data().get();
            for (size_t i = 0; i < factor_descriptors.size(); i++) {
                factor_descriptors[i]->visit_Jtv(visitor, v2.data().get(), v1_ptr);
                v1_ptr += factor_descriptors[i]->get_residual_size();
            }
            cudaDeviceSynchronize();

            // 3. Finally r = b - v2
            saxpy(dim_h, r.data().get(), -1.0, (const T*)v2.data().get(), b);

            for (size_t i = 0; i < max_iter; i++) {

                thrust::fill(v1.begin(), v1.end(), 0);
                // for (size_t i = 0; i < factor_descriptors.size(); i++) {
                //     thrust::fill(v1[i].begin(), v1[i].end(), 0);
                // }




            }
                
        
        
        }   



    };

}