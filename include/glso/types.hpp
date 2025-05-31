#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace glso {

    // using ghalf = __nv_bfloat16;
    using ghalf = __half;
}