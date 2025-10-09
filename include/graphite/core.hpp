#pragma once
#include <graphite/optimizer.hpp>

namespace graphite {

void initialize_cuda() {
  cudaSetDevice(0);
  cudaFree(0);
}

void cleanup_cuda() { cudaDeviceReset(); }

} // namespace graphite
