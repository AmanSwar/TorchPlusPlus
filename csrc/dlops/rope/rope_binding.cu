#include <cuda_runtime.h>
#include <torch/extension.h>


#include "apply_rope.cuh"

void rope_apply_half2(half *x, half *out, half *cos, half *sin, int B, int H,
                      int N, int D);

torch::Tensor rope_apply_cuda(
  torch::Tensor x, torch::Tensor cos_cache,
  torch::Tensor sin_cache
){

  //checks (applied by chatGPT)
  TORCH_CHECK(x.device().is_cuda(), "Input tensor must be on CUDA device");
  TORCH_CHECK(cos_cache.device().is_cuda(), "cos_cache tensor must be on CUDA device");
  TORCH_CHECK(sin_cache.device().is_cuda(), "sin_cache tensor must be on CUDA device");

  TORCH_CHECK(x.dtype() == torch::kFloat16, "Input tensor must be float16");
  TORCH_CHECK(cos_cache.dtype() == torch::kFloat16, "cos_cache tensor must be float16");
  TORCH_CHECK(sin_cache.dtype() == torch::kFloat16, "sin_cache tensor must be float16");

  TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(cos_cache.is_contiguous(), "cos_cache tensor must be contiguous");
  TORCH_CHECK(sin_cache.is_contiguous(), "sin_cache tensor must be contiguous");

  // tensor dims
  auto sizes = x.sizes();
  TORCH_CHECK(sizes.size() == 4,
              "Input tensor must be 4D [batch, heads, seq_len, head_dim]");

  int B = sizes[0]; // bs
  int H = sizes[1]; //num_head
  int N = sizes[2]; //seq_len
  int D = sizes[3]; //head_dim

  TORCH_CHECK(D % 2 == 0, "head dim must be even");

  //validate cos/sin cache dimensions
  auto cos_sizes = cos_cache.sizes();
  auto sin_sizes = sin_cache.sizes();

  TORCH_CHECK(cos_sizes.size() == 2,
              "cos_cache must be 2D [seq_len, head_dim]");
  TORCH_CHECK(sin_sizes.size() == 2,
              "sin_cache must be 2D [seq_len, head_dim]");
  TORCH_CHECK(cos_sizes[0] == N && cos_sizes[1] == D,
              "cos_cache shape mismatch");
  TORCH_CHECK(sin_sizes[0] == N && sin_sizes[1] == D,
              "sin_cache shape mismatch");

  auto output = torch::zeros_like(x);

  half *x_ptr = reinterpret_cast<half *>(x.data_ptr<at::Half>());
  half *out_ptr = reinterpret_cast<half *>(output.data_ptr<at::Half>());
  half *cos_ptr = reinterpret_cast<half *>(cos_cache.data_ptr<at::Half>());
  half *sin_ptr = reinterpret_cast<half *>(sin_cache.data_ptr<at::Half>());

  rope_apply_half2(x_ptr, out_ptr, cos_ptr, sin_ptr, B, H, N, D);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_apply_cuda", &rope_apply_cuda, "RoPE apply CUDA kernel",
        py::arg("x"), py::arg("cos_cache"), py::arg("sin_cache"));
}