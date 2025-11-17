#pragma once
#include <cuda_runtime.h>

namespace graphite {

class StreamPool {
public:
  StreamPool(size_t num_streams)
      : num_streams(num_streams), cleanup_streams(true) {
    streams = new cudaStream_t[num_streams];
    for (size_t i = 0; i < num_streams; ++i) {
      cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
  }

  StreamPool(cudaStream_t *p_streams, size_t n)
      : streams(p_streams), num_streams(n), cleanup_streams(false) {}

  ~StreamPool() {
    if (!cleanup_streams) {
      return;
    }
    for (size_t i = 0; i < num_streams; ++i) {
      cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
  }

  cudaStream_t &select(size_t index) { return streams[index % num_streams]; }

  void sync_all() {
    for (size_t i = 0; i < num_streams; ++i) {
      cudaStreamSynchronize(streams[i]);
    }
  }

  void sync_n(size_t n) {
    n = std::min(n, num_streams);
    for (size_t i = 0; i < n; ++i) {
      cudaStreamSynchronize(streams[i]);
    }
  }

  cudaStream_t *streams;
  size_t num_streams;
  bool cleanup_streams;
};

StreamPool create_default_stream_pool() {
  static cudaStream_t default_stream = cudaStreamPerThread;
  static StreamPool default_pool(&default_stream, 1);
  return default_pool;
}

} // namespace graphite