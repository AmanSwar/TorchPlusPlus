
/*
- SiLU
- GeLU
- GeLU-tanh
- FATReLU
- SwiGLU
*/

#include <cmath>
#include <cuda_runtime.h>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/detail/helper_macros.hpp>
#include <cutlass/half.h>

#include "../dtypes_util.cuh"
#include "../math_util.h"

// Currently only supporting fp16 as majorly all inference is in fp16

namespace torchpp {

namespace activation{

  struct GeluHalf {
  CUTLASS_HOST_DEVICE
  cutlass::half_t operator()(cutlass::half_t const &scaler) const {
    // convert scaler (half) -> float
    float x = float(scaler);
    float const kAlpha = T_SQRT_2_PI; // sqrt(2 / pi)
    float const kBeta = 0.044715;

    float xCubed = x * x * x;
    float tanhArgs = kAlpha * (x + kBeta * xCubed);

    float tanhVal = tanhf(tanhArgs);

    float result = 0.5f * x * (1.0f + tanhVal);

    return cutlass::half_t(result);
  }
};

struct SiluHalf {

  CUTLASS_HOST_DEVICE // __forceinline__ __device__ __host__
      CUTLASSFP16
      operator()(CUTLASSFP16 const &input) const {

    float inputFloat = (float)(input);
    float denom = 1.0f + expf(-inputFloat);
    float result = inputFloat / denom;
    return (CUTLASSFP16)result;
  }
};


struct ReluHalf{
  
  CUTLASS_HOST_DEVICE
  cutlass::half_t operator()(cutlass::half_t const &input) const {
    float inputFloat = float(input);
    float result = fmaxf(0.0f , inputFloat);
    return cutlass::half_t(result);
  }
};


} // namespace activation

} // namespace torchpp