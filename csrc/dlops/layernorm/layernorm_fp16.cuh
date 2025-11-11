#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <cutlass/gemm/threadblock/index_remat.h>
#include <cuda_fp16.h>


#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
  const unsigned int MASK = 0xffffffffu;
#pragma unroll
  for (int offset = (WARP_SIZE >> 1); offset >= 1; offset >>= 1) {
    v += __shfl_xor_sync(MASK, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_reduce_sum_f32(float *sdata, float val) {
  int tid = threadIdx.x;
  int lane = tid % WARP_SIZE;
  int wid = tid / WARP_SIZE; // warp id

  val = warp_reduce_sum_f32(val);

  if (lane == 0)
    sdata[wid] = val;

  __syncthreads();

  float total = 0.0f;
  if (wid == 0) {
    int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    float v = (lane < numWarps) ? sdata[lane] : 0.0f;
    v = warp_reduce_sum_f32(v);
    if (lane == 0)
      sdata[0] = v;
  }

  __syncthreads();
  total = sdata[0];
  return total;
}

__global__ void __launch_bounds__(256 , 2)
layernorm_fp16x2_kernel(
  const half2* __restrict__ inputPtr,
  const half2* __restrict__ weightPtr,
  half2* __restrict__ outputPtr,
  int M , int N, 
  float eps = 1e-6
){

  int tid = threadIdx.x;
  int row = blockIdx.x;

  if(row >= M) return;

  int vcols = (N + 1) / 2; // since each thread  is accessing 2 ele
  int rowStart = row * vcols;

  extern __shared__ char sharedMem[]; //using char cuz sizeof(char) = 1byte

  half2 *smem = reinterpret_cast<half2*>(sharedMem);

  //partial sum smem -> float
  float* smemParitalSum = reinterpret_cast<float*>(smem + vcols);

  float* smemSqSum = reinterpret_cast<float*>(smem + vcols + blockDim.x + 1);

  float partial = 0.0f; // register to hold sum of 2 elem of half2
  float partialSq = 0.0f; // register to hold sum of sq of both elements of half2

  int fullPairs = N / 2; // total number of 2 pairs of elements

  bool hasTail = (N % 2) != 0; // check if a single element is there which is not covered by half2

  for(int idx = tid ; idx  < fullPairs ; idx += blockDim.x){

    half2 element = inputPtr[rowStart + idx];

    float x = __half2float(element.x);
    float y = __half2float(element.y);
    partial += x + y;
    partialSq += x * x + y * y;
    smem[idx] = element;
  }


  //now if it has an element remaining
  if(hasTail){

    int tailIdx = fullPairs;
    if(tid == 0){
      half tailElem = reinterpret_cast<const half* >(inputPtr)[row *N + N - 1];
      float x = __half2float(tailElem);

      partial += x;
      partialSq += x * x;

      half2 newElement;
      newElement.x = tailElem;
      newElement.y = __float2half(0.0f);
      smem[tailIdx] = newElement;
    }
  }


  float totalSum = block_reduce_sum_f32(smemParitalSum, partial);
  float totalSumSq = block_reduce_sum_f32(smemSqSum, partialSq);

  float mean = totalSum / N;
  float var  = (totalSumSq / float(N)) - (mean*mean);\
  float rstd = rsqrt(var + eps);

  //now apply the main formula
  for(int idx = tid ; idx < fullPairs ; idx += blockDim.x){
      half2 inputElement = smem[idx];
      half2 weightElement = weightPtr[idx];
      
      float x = __half2float(inputElement.x);
      float y = __half2float(inputElement.y);

      float outX = ((x - mean) * rstd) * __half2float(weightElement.x);
      float outY = ((y - mean) * rstd) * __half2float(weightElement.y);

      half2 storeElement;
      storeElement.x = __float2half(outX);
      storeElement.y = __float2half(outY);
      outputPtr[rowStart + idx] = storeElement;
  }

  // Handle tail element separately
  if(hasTail && tid == 0){
      half2 inputElement = smem[fullPairs];
      half weightElement = reinterpret_cast<const half*>(weightPtr)[N-1];
      
      float x = __half2float(inputElement.x);
      float outX = ((x - mean) * rstd) * __half2float(weightElement);
      
      reinterpret_cast<half*>(outputPtr)[row * N + N - 1] = __float2half(outX);
  } 
}


void launchLayerNorm(
  const half* inputPtr,
  const half* weightPtr,
  half* outPtr,
  int M , int N , float eps = 1e-6f
){

  int threadPerBlock = 256;
  int blockPerGrid = M;

  int vcols = (N + 1) / 2;
  int numWarps = (threadPerBlock + WARP_SIZE - 1) / WARP_SIZE;
  size_t cacheSize = vcols * sizeof(half2);
  size_t reductionSize = numWarps * sizeof(float);
  size_t reductionSqSize =numWarps  * sizeof(float);

  size_t smemSize = cacheSize + reductionSize + reductionSqSize;

  const half2 *inputPtr2 = reinterpret_cast<const half2 *>(inputPtr);
  const half2 *weightPtr2 = reinterpret_cast<const half2 *>(weightPtr);
  half2 *outPtr2 = reinterpret_cast<half2 *>(outPtr);

  layernorm_fp16x2_kernel<<<blockPerGrid , threadPerBlock , smemSize>>>(
    inputPtr2,
    weightPtr2,
    outPtr2,
    M , N , eps
  );

}


