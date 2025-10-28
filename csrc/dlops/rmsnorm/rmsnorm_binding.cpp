#include "ATen/core/TensorBody.h"
#include "c10/core/ScalarType.h"
#include "c10/util/BFloat16.h"

#include "c10/util/Half.h"
#include "rmsnorm.cuh"

#include <cassert>
#include <torch/extension.h>


extern "C" void launch_rmsnorm_fp16(const half *input_matrix,
                                    const half *weight_matrix, half *out_matrix,
                                    int M, int N, float eps) {
  launch_rmsnorm_fp16_vectorized(input_matrix, weight_matrix, out_matrix, M, N,
                                 eps);
}



torch::Tensor fused_rmsnorm(torch::Tensor input_matrix, torch::Tensor weight,
                            float eps = 1e-6f) {
  TORCH_CHECK(input_matrix.is_cuda(), "input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(input_matrix.scalar_type() == at::kHalf, "input must be half");
  TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight must be half");
  TORCH_CHECK(input_matrix.dim() >= 2,
              "expected input to have >= 2 dims (leading dims..., embed_dim)");

  if (!input_matrix.is_contiguous())
    input_matrix = input_matrix.contiguous();
  if (!weight.is_contiguous())
    weight = weight.contiguous();

  int64_t N64 = input_matrix.size(input_matrix.dim() - 1);
  int64_t M64 = 1;
  for (int i = 0; i < input_matrix.dim() - 1; ++i) {
    M64 *= input_matrix.size(i);
  }

  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == N64,
              "weight must be 1-D and length == embed_dim");

  TORCH_CHECK(M64 > 0 && N64 > 0, "invalid shapes");
  TORCH_CHECK(M64 <= std::numeric_limits<int>::max() &&
                  N64 <= std::numeric_limits<int>::max(),
              "M or N too large for kernel (must fit in 32-bit int)");

  int M = static_cast<int>(M64);
  int N = static_cast<int>(N64);

  auto output = torch::empty_like(input_matrix);

  const half *in_ptr =
      reinterpret_cast<const half *>(input_matrix.data_ptr<at::Half>());
  const half *w_ptr =
      reinterpret_cast<const half *>(weight.data_ptr<at::Half>());
  half *out_ptr = reinterpret_cast<half *>(output.data_ptr<at::Half>());

  // launch kernel with M rows and N columns
  launch_rmsnorm_fp16(in_ptr, w_ptr, out_ptr, M, N, eps);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "Kernel launch failed: ", cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("rmsnorm_kernel_vec", &fused_rmsnorm, "Fused RMSNorm (BF16)");
  m.def("rmsnorm_kernel_vectorized", &fused_rmsnorm, "Fused RMSNorm (BF16)");
}