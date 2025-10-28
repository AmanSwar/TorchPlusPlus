
/*
- SiLU
- GeLU
- GeLU-tanh
- FATReLU
- SwiGLU
*/

#include <__clang_cuda_runtime_wrapper.h>
#include <cmath>
#include <cuda_runtime.h>

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
__device__ __forceinline__ Dtype gelu_tanh_(const Dtype &input){

  const float f = (float)input;
  constexpr float BETA = T_I_SQRT_2 * T_2_I_SQRT_PI * 0.5f;
  constexpr float KAPPA = 0.044715;
  
  float inputCube = input * input * input;

  float innerTerm = BETA * (f + KAPPA * inputCube);

  return (Dtype)(0.5f * f * (1.0f + tanhf(innerTerm)));
}



}