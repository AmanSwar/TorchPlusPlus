#pragma once

#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cute/arch/copy.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/int.hpp>
#include <cute/pointer.hpp>
#include <cute/stride.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_impl.hpp>
#include <stdlib.h>

#include "../../utils.cuh"

// ======================================== fp16x8 ===========================
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
      //kDim -> each element will be processing 4 elemets so stride of 4
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


// =============================== CuTe ==========================================

using namespace cute;

template<
  typename TiledCopy,
  int BlockM,
  int BlockK,
  int NumElementsPerThread
>
__global__ void hgemvCuteF16x8Kernel(
  half* aPtr, // input matrix
  half* xPtr, // input vector
  half* yPtr, // output vector
  const int M , 
  const int K
){

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    int bix = blockIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    auto aTensor = make_tensor(
      make_gmem_ptr(aPtr),
      make_layout(
        make_shape(M , K),
        make_stride(K , Int<1>{})
      )
    );

    auto xTensor = make_tensor(
      make_gmem_ptr(xPtr),
      make_layout(
        make_shape(M , K),
        make_stride(0 , Int<1>{})
      )
    );

    auto yTensor = make_tensor(
      make_gmem_ptr(yPtr),
      make_layout(
        make_shape(M , 1),
        make_stride(Int<1>{} , 0)
      )
    );


    auto axPre = make_identity_tensor(shape(aTensor));
    auto yPre = make_identity_tensor(shape(yTensor));

    auto gA = local_tile(
      aTensor,
      make_shape(Int<BlockM>{} , Int<BlockK>{}),
      make_coord(bix , _)
    );

    auto gX = local_tile(
      xTensor,
      make_shape(Int<BlockM>{} , Int<BlockK>{}),
      make_coord(bix , _)
    );

    auto gY = local_tile(
      yTensor,
      make_shape(Int<BlockM>{} , Int<1>{}),
      make_coord(bix , 0)
    );

    auto gAXPre = local_tile(
      axPre ,
      make_shape(Int<BlockM>{} , Int<BlockK>{}),
      make_coord(bix , _)
    );

    auto gYPre = local_tile(
      yPre,
      make_shape(Int<BlockM>{} , Int<1>{}),
      make_coord(bix , _) 
    );

    TiledCopy tiledCopy;

    auto thr_copy = tiledCopy.get_slice(tid);

    auto tAgA = thr_copy.partition_S(gA);
    auto tXgX = thr_copy.partition_S(gX);
    auto rAXPre = thr_copy.partition_S(gAXPre);

    const int numTileK = size<2>(gA);

    auto tArA = make_tensor_like(tAgA(_, _, _, 0));
    auto tXrX = make_tensor_like(tXgX(_, _, _, 0));

    auto sum = make_tensor_like(gY(0, _));
    clear(sum);

  #pragma unroll
  for (int iterK = 0; iterK < numTileK; iterK++) {
    auto pre_ = rAXPre(_, _, _, iterK);
    auto pred = [&](auto... coords) {
      return cute::elem_less(pre_(NumElementsPerThread - 1), shape(aTensor));
    };

    clear(tArA);
    copy_if(tiledCopy, pred, tAgA(_, _, _, iterK), tArA);
    clear(tXrX);
    copy_if(tiledCopy, pred, tXgX(_, _, _, iterK), tXrX);

    auto tArA_half2 = recast<half2>(tArA);
    auto tXrX_half2 = recast<half2>(tXrX);
    auto sum_half2 = make_tensor<half2>(make_shape(Int<1>{}));

  
    #pragma unroll
    for (int iterElem = 0; iterElem < size(tArA_half2); iterElem++) {
      sum_half2(0) = tArA_half2(iterElem) * tXrX_half2(iterElem) + sum_half2(0);
    }

    sum(0) += sum_half2(0).x + sum_half2(0).y;

  }

  sum(0) = warpReduceSumF16AccF32<WARP_SIZE>(sum(0));

  auto stord_pred = [&](auto... coords) {
    return cute::elem_less(gYPre(warpId), shape(yTensor)) && laneId == 0;
  };

  copy_if(stord_pred, sum, gY(warpId, _));
}


void launchHgemvCute(
  half* inputMatrix,
  half* inputVector,
  half* outputVector,
  int M , int K
){
  constexpr int NUM_THREAD_PER_ROW = 32;
  constexpr int NUM_THREAD_PER_BLOCK =  128;
  constexpr int NUM_ROW_PER_BLOCK = NUM_THREAD_PER_BLOCK / 32;

  using LoadType = uint128_t;

  constexpr int NUM_ELE_PER_THREAD = sizeof(LoadType) / sizeof(half);


  using CopyAtom = Copy_Atom<UniversalCopy<LoadType> , half>;

  using TiledCopy = decltype(
    make_tiled_copy(
      CopyAtom{},
      make_layout(
        Shape<Int<NUM_ROW_PER_BLOCK>, Int<NUM_THREAD_PER_BLOCK>>{},
        GenRowMajor{} 
      ),
      make_layout(Shape<_1 , Int<NUM_ELE_PER_THREAD>>{} , GenRowMajor{})
    )
  );


  dim3 blockSize(NUM_THREAD_PER_ROW , NUM_ROW_PER_BLOCK);
  dim3 gridSize(ceil_div(M , NUM_ROW_PER_BLOCK));


  hgemvCuteF16x8Kernel<TiledCopy , NUM_ROW_PER_BLOCK , NUM_THREAD_PER_ROW * NUM_ELE_PER_THREAD , NUM_ELE_PER_THREAD>
  <<<gridSize , blockSize>>>(
    inputMatrix,
    inputVector,
    outputVector,
    M , K
  )

}