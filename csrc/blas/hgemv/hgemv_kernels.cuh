#pragma once

#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../../utils.cuh"


__global__ void hgemvK128F16x4Kernel(
  half* a, // input matrix (M x N)
  half* x, // input vector (N x 1)
  half* y, // output vector (M x 1)
  int M, 
  int K
){

  int tix = threadIdx.x;
  int tiy = threadIdx.y;

  int bix = blockIdx.x;

  int laneId = tix % WARP_SIZE;

  int mDim = bix * blockDim.y + tiy;

  if(mDim < M){
    half sum  = 0.0f;

    // we are going to process 64bits together -> 4 elements of fp16 -> 2 x half
    // so each warp (32 threads) is processing 32 * 4 elements
    const int ELEMENTS_PER_WARP = WARP_SIZE * 4;
    
    //total num of warps required in kDim -> 
    const int NUM_WARPS = (K + ELEMENTS_PER_WARP - 1) / ELEMENTS_PER_WARP;

    #pragma unroll
    for(int w = 0 ; w < NUM_WARPS ; w++){
      int kDim = (w * WARP_SIZE + laneId) * 4;

      half2 regX0 = HALF2(x[kDim + 0]);
      half2 regX1 = HALF2(x[kDim + 2]);
      half2 regA0 = HALF2(a[mDim * K + kDim + 0]);
      half2 regA1 = HALF2(a[mDim * K + kDim + 2]);

      half2 dp0 = __hmul2(regA0 , regX0);
      half2 dp1 = __hmul2(regA1 , regX1);
      sum += dp0.x + dp0.y + dp1.x + dp1.y;
    }

    sum = warpReduceSumF16AccF32<WARP_SIZE>(sum);

    if(laneId == 0){
      y[mDim] = sum;
    }
  }
}

void launchHgemv(
  half* inputMatrix, 
  half* inputVector, 
  half* outputVector,
  int M , int N
){
  dim3 blockSize(32 , 4);
  dim3 gridSize((M + 4 - 1) / 4);

  hgemvK128F16x4Kernel<<<gridSize , blockSize>>>(
    inputMatrix,
    inputVector,
    outputVector,
    M , N
  );
  
}