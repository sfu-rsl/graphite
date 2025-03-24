#pragma once
#include <glso/common.hpp>
#include <glso/factor.hpp>

namespace glso {

    template<typename T>
    class PCGSolver {
        private:

        thrust::device_vector<T> v;
        
        // Need vectors for residuals of each factor
        std::vector<thrust::device_vector<T>> v1; // Jv
        thrust::device_vector<T> v2; // J^T v1

        thrust::device_vector<T> xout;

        public:

        void solve(GraphVisitor<T>& visitor, 
            std::vector<BaseVertexDescriptor<T>*>& vertex_descriptors,
            std::vector<BaseFactorDescriptor<T>*>& factor_descriptors,
            T* x0, size_t h, size_t max_iter) {

            v1.resize(factor_descriptors.size());
            for (size_t i = 0; i < factor_descriptors.size(); i++) {
                v1[i].resize(factor_descriptors[i]->get_residual_size());
            }


            v2.resize(h);

            thrust::fill(v2.begin(), v2.end(), 0);

            xout.resize(h);
            thrust::fill(xout.begin(), xout.end(), 0);

            for (size_t i = 0; i < max_iter; i++) {

                for (size_t i = 0; i < factor_descriptors.size(); i++) {
                    thrust::fill(v1[i].begin(), v1[i].end(), 0);
                }

            }
                
        
        
        }   



    };

}