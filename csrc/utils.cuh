#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

//constants
#define WARP_SIZE 32


// dtype conversions
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])


// load intrinsics
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])



//load matrix

//load one matrix of size 8x8 and dtype = half from shared mem (addr) to registers (R)
#define LDMATRIX_X1(R, addr)\
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" \
               : "=r"(R) \
               : "r"(addr))

//load 2 matrix of size 8x8 (half) from shared mem to registers (128 elements = 256 bytes)
#define LDMATRIX_X2(R0, R1, addr)\
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"\
               : "=r"(R0), "=r"(R1)\
               : "r"(addr))

// load 4 tiles of size 8x8 (half) from shared mem to registers
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))

// load one tile but transposed (shared mem (addr) -> registers (R) transposed)
#define LDMATRIX_X1_T(R, addr)                                                 \
  asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"  \
               : "=r"(R)                                                       \
               : "r"(addr))

#define LDMATRIX_X2_T(R0, R1, addr)                                            \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"       \
      : "=r"(R0), "=r"(R1)                                                     \
      : "r"(addr))


#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)                                    \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, "      \
      "[%4];\n"                                                                \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))




// copy async instrincs
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

#define CP_ASYNC_WAIT_GROUP(n)                                                 \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

#define CP_ASYNC_CA(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))

      
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))


      

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