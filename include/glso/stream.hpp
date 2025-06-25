#pragma once
#include <cuda_runtime.h>

namespace glso {

    class StreamPool {
        public:

        StreamPool(size_t num_streams) : num_streams(num_streams) {
            streams = new cudaStream_t[num_streams];
            for (size_t i = 0; i < num_streams; ++i) {
                cudaStreamCreate(&streams[i]);
            }
        }

        ~StreamPool() {
            for (size_t i = 0; i < num_streams; ++i) {
                cudaStreamDestroy(streams[i]);
            }
            delete[] streams;
        }

        cudaStream_t& select(size_t index) {
            return streams[index % num_streams];
        }

        cudaStream_t *streams;
        size_t num_streams;
    };
}