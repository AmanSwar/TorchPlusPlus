#include <cutlass/cutlass.h>
#include <torch/extension.h>

#include "fused_gemm.cuh"

// NOTE : CURRENTLY ONLY SUPPORTING FP16 
// CUZ I AM GPU POOR AND I ONLY HAVE ACCESS TO MY LOCAL RTX 3050
// TENSOR CORES DOESN'T SUPPORT ANY OTHER DTYPE FOR SM80 AMPERE ARCH


torch::Tensor LinearGeluFused(
  torch::Tensor input,
  torch::Tensor weight
){
  TORCH_CHECK(input.dim() == 2, "Input must be 2D");
  TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
  TORCH_CHECK(input.size(1) == weight.size(1), "Dimension mismatch");
  TORCH_CHECK(input.dtype() == torch::kFloat16, "Input must be FP16");
  TORCH_CHECK(weight.dtype() == torch::kFloat16, "Weight must be FP16");
  TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
  TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
  
  int M = input.size(0);
  int K = input.size(1);
  int N = weight.size(0);

  auto output = torch::empty(
    {M, N}, 
    torch::TensorOptions().dtype(torch::kFloat16).device(input.device())
  );

  
  using GemmGelu = typename torchpp::fused_linear::GemmActivation<torchpp::activation::GeluHalf>::Gemm;

  typename GemmGelu::Arguments args{
    {M, N, K},
    {reinterpret_cast<torchpp::fused_linear::ElementA*>(input.data_ptr<at::Half>()), K},
    {reinterpret_cast<torchpp::fused_linear::ElementB*>(weight.data_ptr<at::Half>()), K},
    {reinterpret_cast<torchpp::fused_linear::ElementC*>(output.data_ptr<at::Half>()), N},
    {reinterpret_cast<torchpp::fused_linear::ElementC*>(output.data_ptr<at::Half>()), N},
    {cutlass::half_t(1.0f), cutlass::half_t(0.0f)},
    1
  };

  GemmGelu gemmOp;

  cutlass::Status status = gemmOp.initialize(args);

  status = gemmOp();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS kernel execution failed: " + 
                           std::string(cutlassGetStatusString(status)));
  }
  
  return output;

}

torch::Tensor LinearSiluFused(
  torch::Tensor input,
  torch::Tensor weight
){
  TORCH_CHECK(input.dim() == 2, "Input must be 2D");
  TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
  TORCH_CHECK(input.size(1) == weight.size(1), "Dimension mismatch");
  TORCH_CHECK(input.dtype() == torch::kFloat16, "Input must be FP16");
  TORCH_CHECK(weight.dtype() == torch::kFloat16, "Weight must be FP16");
  TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
  TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
  
  int M = input.size(0);
  int K = input.size(1);
  int N = weight.size(0);

  auto output = torch::empty(
    {M, N}, 
    torch::TensorOptions().dtype(torch::kFloat16).device(input.device())
  );

  
  using GemmGelu = typename torchpp::fused_linear::GemmActivation<torchpp::activation::SiluHalf>::Gemm;

  typename GemmGelu::Arguments args{
    {M, N, K},
    {reinterpret_cast<torchpp::fused_linear::ElementA*>(input.data_ptr<at::Half>()), K},
    {reinterpret_cast<torchpp::fused_linear::ElementB*>(weight.data_ptr<at::Half>()), K},
    {reinterpret_cast<torchpp::fused_linear::ElementC*>(output.data_ptr<at::Half>()), N},
    {reinterpret_cast<torchpp::fused_linear::ElementC*>(output.data_ptr<at::Half>()), N},
    {cutlass::half_t(1.0f), cutlass::half_t(0.0f)},
    1
  };

  GemmGelu gemmOp;

  cutlass::Status status = gemmOp.initialize(args);

  status = gemmOp();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS kernel execution failed: " + 
                           std::string(cutlassGetStatusString(status)));
  }
  
  return output;

}


torch::Tensor LinearReluFused(
  torch::Tensor input,
  torch::Tensor weight
){
  TORCH_CHECK(input.dim() == 2, "Input must be 2D");
  TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
  TORCH_CHECK(input.size(1) == weight.size(1), "Dimension mismatch");
  TORCH_CHECK(input.dtype() == torch::kFloat16, "Input must be FP16");
  TORCH_CHECK(weight.dtype() == torch::kFloat16, "Weight must be FP16");
  TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
  TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
  
  int M = input.size(0);
  int K = input.size(1);
  int N = weight.size(0);

  auto output = torch::empty(
    {M, N}, 
    torch::TensorOptions().dtype(torch::kFloat16).device(input.device())
  );

  
  using GemmGelu = typename torchpp::fused_linear::GemmActivation<torchpp::activation::ReluHalf>::Gemm;

  typename GemmGelu::Arguments args{
    {M, N, K},
    {reinterpret_cast<torchpp::fused_linear::ElementA*>(input.data_ptr<at::Half>()), K},
    {reinterpret_cast<torchpp::fused_linear::ElementB*>(weight.data_ptr<at::Half>()), K},
    {reinterpret_cast<torchpp::fused_linear::ElementC*>(output.data_ptr<at::Half>()), N},
    {reinterpret_cast<torchpp::fused_linear::ElementC*>(output.data_ptr<at::Half>()), N},
    {cutlass::half_t(1.0f), cutlass::half_t(0.0f)},
    1
  };

  GemmGelu gemmOp;

  cutlass::Status status = gemmOp.initialize(args);

  status = gemmOp();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS kernel execution failed: " + 
                           std::string(cutlassGetStatusString(status)));
  }
  
  return output;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("LinearGelu", &LinearGeluFused, "CUTLASS Linear + GELU forward (FP16)");
  m.def("LinearSilu", &LinearSiluFused, "CUTLASS Linear + Silu forward (FP16)");
  m.def("LinearRelu", &LinearReluFused, "CUTLASS Linear + Relu forward (FP16)");
}