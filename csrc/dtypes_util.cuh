#include <cuda_fp16.h>
#include <cutlass/half.h>
#include <cuda_fp8.h>


namespace torchpp{

#define FP32 float
#define FP16 half
#define FP8 __nv_fp8_e4m3 
#define CUTLASSFP16 cutlass::half_t

  
}