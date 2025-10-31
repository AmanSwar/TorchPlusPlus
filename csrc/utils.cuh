#pragma once

#include <cstdio>

#include <__clang_cuda_runtime_wrapper.h>
#include <cstdlib>
#include <cuda_runtime.h>

void __forceinline__ CUDA_CHECK_ERR(cudaError_t error , const char* file , int line){
  if(error != cudaSuccess){
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}