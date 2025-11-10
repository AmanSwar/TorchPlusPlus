#include <complex.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/detail/helper_macros.hpp>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_size.h>

#include "activation.cuh"
#include "../dtypes_util.cuh"

namespace torchpp{

template <
  typename ElemOutputDtype,
  typename ElemAccDtype,
  typename ElemComputeDtype,
  int Count
>
class LinearGeluFP16{
public:
  using ElemOutput = ElemOutputDtype;
  using ElemAcc = ElemAccDtype;
  using ElemCompute = ElemComputeDtype;

  static int const kCount = Count;

  using FragmentOutput = cutlass::Array<ElemOutput , kCount>;
  using FragmentAcc = cutlass::Array<ElemAcc , kCount>;
  using ComputeFragment = cutlass::Array<ElemCompute , kCount>;

  struct Params {
    ElemCompute alpha;
    ElemCompute beta;
    
    // constructor
    CUTLASS_HOST_DEVICE
    Params() : alpha(ElemCompute(1)), beta(ElemCompute(0)) {}

    CUTLASS_HOST_DEVICE
    Params(ElemCompute alpha , ElemCompute beta) : alpha(alpha) , beta(beta) {}
  };

private:
  ElemCompute alpha_;
  ElemCompute beta_;

public:
  CUTLASS_HOST_DEVICE
  LinearGeluFP16(Params const &params){
    alpha_ = params.alpha;
    beta_ = params.beta;
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAcc const &accumulator,
    FragmentOutput const &source
  ) const {

    ComputeFragment convertedAcc;
    ComputeFragment convertedSrc;

    cutlass::NumericArrayConverter<ElemCompute , ElemAcc , kCount> accConverter;
    cutlass::NumericArrayConverter<ElemCompute, ElemOutput, kCount> srcConverter;

    ComputeFragment intermediate;

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0 ; i < kCount ; i++){
      intermediate[i] = alpha_ * convertedAcc[i] + beta_ * convertedSrc[i];
    }

    
    //convert half for GeLU
    cutlass::Array<cutlass::half_t , kCount> halfIntermediate;
    cutlass::NumericArrayConverter<CUTLASSFP16, ElemCompute, kCount> toHalf;
    halfIntermediate = toHalf(intermediate);
    
    GeluHalf gelu;

    cutlass::Array<CUTLASSFP16, kCount> halfResult;

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0 ; i < kCount ; i++){
      halfResult[i] = gelu(halfIntermediate[i]);
    }


    
    //convert back to output type
    cutlass::NumericArrayConverter<ElemOutput, cutlass::half_t, kCount> outputConverter;
    return outputConverter(halfResult);

  
  }

};


using ElementA = CUTLASSFP16;
using ElementB = CUTLASSFP16;
using ElementC = CUTLASSFP16;
using ElementAcc = FP32;
using ElementCompute = CUTLASSFP16;


using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

static int const kAlignA = 8;
static int const kAlignB = 8;

using EpilogueOp = LinearGeluFP16<
  ElementC,
  ElementAcc,
  ElementCompute,
  128 / cutlass::sizeof_bits<ElementC>::value
>;

using Gemm = cutlass::gemm::device::Gemm<
  ElementA , LayoutA,
  ElementB , LayoutB,
  ElementC , LayoutC,
  ElementAcc,
  cutlass::arch::OpClassTensorOp, // forcing it to use tensor cores (only compatible for volta++ arch)
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128 , 128 , 32>,
  cutlass::gemm::GemmShape<64,64,32>,
  cutlass::gemm::GemmShape<16,8,16>,
  EpilogueOp,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,
  kAlignA,
  kAlignB
>;


}