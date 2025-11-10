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
  using ElementOutput = ElemOutputDtype;
  using ElementAccumulator  = ElemAccDtype;
  using ElementCompute = ElemComputeDtype;

  static int const kCount = Count;

  using FragmentOutput = cutlass::Array<ElementOutput , kCount>;
  using FragmentAcc = cutlass::Array<ElementAccumulator , kCount>;
  using ComputeFragment = cutlass::Array<ElementCompute , kCount>;

  struct Params {
    ElementCompute alpha;
    ElementCompute beta;
    
    // constructor
    CUTLASS_HOST_DEVICE
    Params() : alpha(ElementCompute(1)), beta(ElementCompute(0)) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha , ElementCompute beta) : alpha(alpha) , beta(beta) {}
  };

private:
  ElementCompute alpha_;
  ElementCompute beta_;

public:
  CUTLASS_HOST_DEVICE
  LinearGeluFP16(Params const &params){
    alpha_ = params.alpha;
    beta_ = params.beta;
  }

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const{
    return beta_ != ElementCompute(0);
  }
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {}
  

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAcc const &accumulator,
    FragmentOutput const &source
  ) const {

    ComputeFragment convertedAcc;
    ComputeFragment convertedSrc;

    cutlass::NumericArrayConverter<ElementCompute , ElementAccumulator , kCount> accConverter;
    cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount> srcConverter;

    ComputeFragment intermediate;

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0 ; i < kCount ; i++){
      intermediate[i] = alpha_ * convertedAcc[i] + beta_ * convertedSrc[i];
    }

    
    //convert half for GeLU
    cutlass::Array<cutlass::half_t , kCount> halfIntermediate;
    cutlass::NumericArrayConverter<CUTLASSFP16, ElementCompute, kCount> toHalf;
    halfIntermediate = toHalf(intermediate);
    
    GeluHalf gelu;

    cutlass::Array<CUTLASSFP16, kCount> halfResult;

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0 ; i < kCount ; i++){
      halfResult[i] = gelu(halfIntermediate[i]);
    }


    
    //convert back to output type
    cutlass::NumericArrayConverter<ElementOutput, cutlass::half_t, kCount> outputConverter;
    return outputConverter(halfResult);
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAcc const &accumulator
  ) const {
    FragmentOutput source;
    source.clear();
    return (*this)(accumulator, source);
  }

};

namespace fused_linear{
using ElementA = CUTLASSFP16;
using ElementB = CUTLASSFP16;
using ElementC = CUTLASSFP16;
using ElementAcc = FP32;
using ElementCompute = FP32;


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
}