
/*
- SiLU
- GeLU
- GeLU-tanh
- FATReLU
- SwiGLU
*/

#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <utility>

#include "../math_util.h"


namespace torchpp{

template <typename Dtype> 
__device__ __forceinline__ Dtype silu(const Dtype &input){
  return (Dtype)((float)(input) / (1.0f + expf((float)-input)));
} 

template <typename Dtype>
__device__ __forceinline__ Dtype gelu(const Dtype &input){
  const float f = (float)input;
  constexpr float ALPHA = T_I_SQRT_2;
  return (Dtype)(f * 0.5f * (1.0f + erf(f * ALPHA)));
}

template <typename Dtype>
__device__ __forceinline__ Dtype geluTanh(const Dtype &input){

  const float f = (float)input;
  constexpr float BETA = T_I_SQRT_2 * T_2_I_SQRT_PI * 0.5f;
  constexpr float KAPPA = 0.044715;
   
  float inputCube = input * input * input;

  float innerTerm = BETA * (f + KAPPA * inputCube);

  return (Dtype)(0.5f * f * (1.0f + tanhf(innerTerm)));
}



template<
  typename Dtype,
  Dtype (*ACT_FN)(const Dtype&),
  bool actFirst
>
__device__ __forceinline__ Dtype compute(
  const Dtype& x,
  const Dtype& y
){
  //conditional -> if activcation first then actfn(x) * y 
  // else -> x * actfn(y)
  return actFirst ? ACT_FN(x) * y : x * ACT_FN(y); 
}



template<
  typename Dtype,
  Dtype (*ACT_FN)(const Dtype&),
  bool actFirst
>
__global__ void actMulKernel(
  const Dtype* __restrict__ input,
  Dtype* __restrict__ output,
  const int D
){

  const int64_t tokenIdx = blockIdx.x;

  for(int64_t idx = threadIdx.x; idx < D ; idx += blockDim.x){
    const Dtype x = __ldg(&input[tokenIdx * 2 * D + idx]);
    const Dtype y = __ldg(&input[tokenIdx * 2 * D + D + idx]);

    output[tokenIdx * D + idx] = compute<Dtype , ACT_FN , actFirst>(x,y);
  }
}


}