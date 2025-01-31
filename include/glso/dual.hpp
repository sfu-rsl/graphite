#pragma once
#include <cuda/std/limits>

namespace glso {

template <typename T>
struct Dual {
    T real;
    T dual;

    __host__ __device__ Dual() : real(0), dual(0) {}
    __host__ __device__ Dual(T real, T dual) : real(real), dual(dual) {}

    __host__ __device__ Dual<T> operator+(const Dual<T>& other) const {
        return Dual<T>(real + other.real, dual + other.dual);
    }

    __host__ __device__ Dual<T> operator-(const Dual<T>& other) const {
        return Dual<T>(real - other.real, dual - other.dual);
    }

    __host__ __device__ Dual<T> operator*(const Dual<T>& other) const {
        return Dual<T>(real * other.real, real * other.dual + dual * other.real);
    }

    __host__ __device__ Dual<T> operator/(const Dual<T>& other) const {
        if (other.real == 0) {
            // Handle division by zero case
            return Dual<T>(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity());
        }
        T denominator = other.real * other.real;
        return Dual<T>((real * other.real) / denominator, (dual * other.real - real * other.dual) / denominator);
    }

    __host__ __device__ Dual<T>& operator+=(const Dual<T>& other) {
        real += other.real;
        dual += other.dual;
        return *this;
    }

    __host__ __device__ Dual<T>& operator-=(const Dual<T>& other) {
        real -= other.real;
        dual -= other.dual;
        return *this;
    }

    __host__ __device__ Dual<T>& operator*=(const Dual<T>& other) {
        T new_real = real * other.real;
        dual = real * other.dual + dual * other.real;
        real = new_real;
        return *this;
    }

    __host__ __device__ Dual<T>& operator/=(const Dual<T>& other) {
        if (other.real == 0) {
            // Handle division by zero case
            real = std::numeric_limits<T>::infinity();
            dual = std::numeric_limits<T>::infinity();
            return *this;
        }
        T denominator = other.real * other.real;
        T new_real = (real * other.real) / denominator;
        dual = (dual * other.real - real * other.dual) / denominator;
        real = new_real;
        return *this;
    }
};

} // namespace glso
