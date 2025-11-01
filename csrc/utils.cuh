#pragma once

#include <__clang_cuda_runtime_wrapper.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32

#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])


void __forceinline__ CUDA_CHECK_ERR(cudaError_t error , const char* file , int line){
  if(error != cudaSuccess){
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}


template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warpReduceSumF16(half value){
  uint64_t MASK = 0xffffffff;
  #pragma unroll
  for(int offset = kWarpSize >> 1 ; offset >= 1 ; offset >>= 1){
    value += __shfl_xor_sync(MASK , value , offset);
  }

  return value;
}


template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warpReduceSumF16AccF32(half value){
  uint64_t MASK = 0xffffffff;
  float value32 = __half2float(value);
  #pragma unroll
  for(int offset = kWarpSize >> 1 ; offset >= 1 ; offset >>= 1){
    value32 += __shfl_xor_sync(MASK , value32 , offset);
  }
  return __float2half(value32);
}