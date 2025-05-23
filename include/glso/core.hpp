#pragma once
#include <glso/optimizer.hpp>

namespace glso {

void initialize_cuda() {
  cudaSetDevice(0);
  cudaFree(0);
}

void cleanup_cuda() { cudaDeviceReset(); }

} // namespace glso
