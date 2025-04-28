#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <Eigen/Dense>
#include <chrono>

using Eigen::Matrix3d;

// Kernel for Structure of Arrays (SoA)
__global__ void multiplySoA(const double* A, const double* B, const double* C, const double* D, double* output, int numItems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numItems) {
        Matrix3d matA, matB, matC, matD, result;

        // Load matrices from SoA
        for (int i = 0; i < 9; ++i) {
            matA(i / 3, i % 3) = A[idx * 9 + i];
            matB(i / 3, i % 3) = B[idx * 9 + i];
            matC(i / 3, i % 3) = C[idx * 9 + i];
            matD(i / 3, i % 3) = D[idx * 9 + i];
        }

        // Perform multiplication
        result = matA * matB * matC * matD;

        // Store result
        for (int i = 0; i < 9; ++i) {
            output[idx * 9 + i] = result(i / 3, i % 3);
        }
    }
}

// Kernel for Array of Structs (AoS)
struct Matrices {
    double A[9];
    double B[9];
    double C[9];
    double D[9];
};

__global__ void multiplyAoS(const Matrices* input, double* output, int numItems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numItems) {
        Matrix3d matA, matB, matC, matD, result;

        // Load matrices from AoS
        for (int i = 0; i < 9; ++i) {
            matA(i / 3, i % 3) = input[idx].A[i];
            matB(i / 3, i % 3) = input[idx].B[i];
            matC(i / 3, i % 3) = input[idx].C[i];
            matD(i / 3, i % 3) = input[idx].D[i];
        }

        // Perform multiplication
        result = matA * matB * matC * matD;

        // Store result
        for (int i = 0; i < 9; ++i) {
            output[idx * 9 + i] = result(i / 3, i % 3);
        }
    }
}

int main() {
    const int numItems = 1e6;
    const int matrixSize = 9; // 3x3 matrix
    const int totalSize = numItems * matrixSize;

    // Allocate and initialize host memory using thrust::host_vector
    thrust::host_vector<double> h_A(totalSize), h_B(totalSize), h_C(totalSize), h_D(totalSize);
    thrust::host_vector<Matrices> h_inputAoS(numItems);
    thrust::host_vector<double> h_outputSoA(totalSize), h_outputAoS(totalSize);

    // Initialize matrices with random values
    for (int i = 0; i < totalSize; ++i) {
        h_A[i] = static_cast<double>(rand()) / RAND_MAX;
        h_B[i] = static_cast<double>(rand()) / RAND_MAX;
        h_C[i] = static_cast<double>(rand()) / RAND_MAX;
        h_D[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < numItems; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            h_inputAoS[i].A[j] = h_A[i * matrixSize + j];
            h_inputAoS[i].B[j] = h_B[i * matrixSize + j];
            h_inputAoS[i].C[j] = h_C[i * matrixSize + j];
            h_inputAoS[i].D[j] = h_D[i * matrixSize + j];
        }
    }

    // Allocate device memory using thrust::device_vector
    thrust::device_vector<double> d_A = h_A, d_B = h_B, d_C = h_C, d_D = h_D, d_outputSoA(totalSize), d_outputAoS(totalSize);
    thrust::device_vector<Matrices> d_inputAoS = h_inputAoS;

    // Launch kernels and time them
    const int threadsPerBlock = numItems;
    const int blocksPerGrid = (numItems + threadsPerBlock - 1) / threadsPerBlock;

    double totalTimeSoA = 0.0, totalTimeAoS = 0.0;
    const int numRuns = 1000;

    for (int run = 0; run < numRuns; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        multiplySoA<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(d_A.data()),
                                                        thrust::raw_pointer_cast(d_B.data()),
                                                        thrust::raw_pointer_cast(d_C.data()),
                                                        thrust::raw_pointer_cast(d_D.data()),
                                                        thrust::raw_pointer_cast(d_outputSoA.data()), numItems);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        totalTimeSoA += std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        multiplyAoS<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(d_inputAoS.data()),
                                                        thrust::raw_pointer_cast(d_outputAoS.data()), numItems);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        totalTimeAoS += std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::cout << "Average time for SoA kernel: " << (totalTimeSoA / numRuns) << " ms" << std::endl;
    std::cout << "Average time for AoS kernel: " << (totalTimeAoS / numRuns) << " ms" << std::endl;

    // Copy results back to host
    thrust::copy(d_outputSoA.begin(), d_outputSoA.end(), h_outputSoA.begin());
    thrust::copy(d_outputAoS.begin(), d_outputAoS.end(), h_outputAoS.begin());

    return 0;
}
