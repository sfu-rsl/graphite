#pragma once
#include <glso/vertex.hpp>
#include <glso/factor.hpp>

namespace glso {

template<typename T>
class Preconditioner {

    virtual void precompute(
        std::vector<BaseVertexDescriptor<T>*>& vertex_descriptors,
            std::vector<BaseFactorDescriptor<T>*>& factor_descriptors, size_t dimension) = 0;

    virtual void apply(T* z, const T* r) = 0;
};

template<typename T>
class IdentityPreconditioner: public Preconditioner<T> {
    private:
        size_t dimension;
    public:
        void precompute(
            std::vector<BaseVertexDescriptor<T>*>& vertex_descriptors,
            std::vector<BaseFactorDescriptor<T>*>& factor_descriptors, size_t dimension) override {
                this->dimension = dimension;
            }

        void apply(T* z, const T* r) override {
            // thrust::copy(r, r + dimension, z);
            cudaMemcpy(z, r, dimension*sizeof(T), cudaMemcpyDeviceToDevice);
        }

};

}