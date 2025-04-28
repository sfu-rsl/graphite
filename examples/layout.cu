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
        // Use Eigen Maps to directly map input data to Eigen matrices
        Eigen::Map<const Matrix3d> matA(&A[idx * 9]);
        Eigen::Map<const Matrix3d> matB(&B[idx * 9]);
        Eigen::Map<const Matrix3d> matC(&C[idx * 9]);
        Eigen::Map<const Matrix3d> matD(&D[idx * 9]);

        // Perform multiplication
        Eigen::Map<Matrix3d> result(&output[idx * 9]);
        result = matA * matB * matC * matD;

        // Store result using Eigen Map
        // Eigen::Map<Matrix3d>(&output[idx * 9]) = result;
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
        // Use Eigen Maps to directly map input data to Eigen matrices
        Eigen::Map<const Matrix3d> matA(input[idx].A);
        Eigen::Map<const Matrix3d> matB(input[idx].B);
        Eigen::Map<const Matrix3d> matC(input[idx].C);
        Eigen::Map<const Matrix3d> matD(input[idx].D);

        // Perform multiplication
        Eigen::Map<Matrix3d> result(&output[idx * 9]);
        result = matA * matB * matC * matD;

        // Store result using Eigen Map
        // Eigen::Map<Matrix3d>(&output[idx * 9]) = result;
    }
}

int main() {
    const int numItems = 2e6;
    const int matrixSize = 9; // 3x3 matrix
    const int totalSize = numItems * matrixSize;

    // Allocate and initialize host memory using thrust::host_vector
    thrust::host_vector<double> h_A(totalSize), h_B(totalSize), h_C(totalSize), h_D(totalSize);
    thrust::host_vector<Matrices> h_inputAoS(numItems);
    thrust::host_vector<double> h_outputSoA(totalSize), h_outputAoS(totalSize);

    // Initialize matrices with random values
    std::vector<Matrix3d> vecA(numItems), vecB(numItems), vecC(numItems), vecD(numItems);

    for (int i = 0; i < numItems; ++i) {
        vecA[i] = Matrix3d::Random();
        vecB[i] = Matrix3d::Random();
        vecC[i] = Matrix3d::Random();
        vecD[i] = Matrix3d::Random();
    }

    for (int i = 0; i < numItems; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            h_A[i * matrixSize + j] = vecA[i](j / 3, j % 3);
            h_B[i * matrixSize + j] = vecB[i](j / 3, j % 3);
            h_C[i * matrixSize + j] = vecC[i](j / 3, j % 3);
            h_D[i * matrixSize + j] = vecD[i](j / 3, j % 3);
        }
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
    const int numRuns = 10000;

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
