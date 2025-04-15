#pragma once
#include <thrust/device_vector.h>


namespace glso {


    template <typename T>
    void print_device_vector(const thrust::device_vector<T>& vec) {
        std::vector<T> host_vec(vec.size());
        thrust::copy(vec.begin(), vec.end(), host_vec.begin());
        for (const auto& val : host_vec) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    template <typename T>
    void print_device_vector(const T* vec, size_t size) {
        std::vector<T> host_vec(size);
        cudaMemcpy(host_vec.data(), vec, size * sizeof(T), cudaMemcpyDeviceToHost);
        for (const auto& val : host_vec) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

}