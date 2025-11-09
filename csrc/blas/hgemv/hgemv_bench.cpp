#include "c10/util/irange.h"
#include "hgemv_kernels.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>

#include <iostream>

void hgemvCpu(
  half* inputMatrix,
  half* inputVector,
  half* outputVector,
  int M , int K
){

  for(int row = 0 ; row < M ; row++){
    float acc = 0;
    for(int col = 0 ; col < K ; col ++){
      acc += __half2float(inputMatrix[row * K + col] * inputVector[col]);
    }
    outputVector[row] = acc;
  }
}



void verify(
  half* kernelOutput, // dim (M, 1)
  half* cpuOutput, // dim (M , 1)
  int M 
){

  for(int row = 0 ; row < M  ; row ++){
    float diff = __half2float(kernelOutput[row] - cpuOutput[row]);
    if(std::abs(diff) > 1e-3){
      std::cout << "Difference of " <<  std::abs(diff) << "\n";
      std::cout << "Mismatch at index : " << row  << "\n";
      return;
    }
  }

  std::cout << "Pass";
  return; 
}



void benchmark(
  void(*function)(half* , half* , half* , int , int) ,
  int M , 
  int N ,
  const char* &functionName
){

  half* input_matrix , *input_vector, *output_vector;

  cudaMalloc(&input_matrix , sizeof(half) * M * N);
  cudaMalloc(&input_vector , sizeof(half)*N);
  cudaMalloc(&output_vector , sizeof(half) * M);
  cudaEvent_t start , end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);


}

void inline initArray(half* arr , int M){
  for(int i = 0 ; i < M ; i++){
    arr[i] = i+1;
  }
}



int main(){

  const int M = 1024;
  const int K = 2048;

  half* dA, *dX , *dY , *cpuOut;

  dA = new half[M * K];
  dX = new half[K];
  dY = new half[M];
  cpuOut = new half[M];

  initArray(dA, M*K);
  initArray(dX , K);
  
  hgemvCpu(dA, dX, cpuOut, M, K);

  half* cA , *cX , *kernelOut;

  


}