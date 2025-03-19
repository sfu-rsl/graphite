#pragma once
#include <glso/common.hpp>
#include <glso/factor.hpp>

namespace glso {

    template<typename T>
    class PCGSolver {
        private:

        thrust::device_vector<T> v;

        public:

        void solve(GraphVisitor& visitor, 
            std::vector<BaseVertexDescriptor<T>*>& vertex_descriptors,
            std::vector<BaseFactorDescriptor<T>*>& factor_descriptors,
            T* x1, T* x0, size_t max_iter) {


            for (size_t i = 0; i < max_iter; i++) {


            }
                
        
        
        }   



    };

}