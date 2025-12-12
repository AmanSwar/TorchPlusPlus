#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "qkrope.cuh"


void rope_apply_half2_dual(
    half *q_in, half *k_in,
    half *q_out, half *k_out,
    half *cos, half *sin,
    int B, int H, int N, int D);

std::vector<torch::Tensor> rope_apply_cuda_dual(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos_cache,
    torch::Tensor sin_cache)
{
    // Input validation for query
    TORCH_CHECK(q.device().is_cuda(), "Query tensor must be on CUDA device");
    TORCH_CHECK(q.dtype() == torch::kFloat16, "Query tensor must be float16");
    TORCH_CHECK(q.is_contiguous(), "Query tensor must be contiguous");
    
    // Input validation for key
    TORCH_CHECK(k.device().is_cuda(), "Key tensor must be on CUDA device");
    TORCH_CHECK(k.dtype() == torch::kFloat16, "Key tensor must be float16");
    TORCH_CHECK(k.is_contiguous(), "Key tensor must be contiguous");
    
    // Input validation for cos/sin cache
    TORCH_CHECK(cos_cache.device().is_cuda(), "cos_cache tensor must be on CUDA device");
    TORCH_CHECK(sin_cache.device().is_cuda(), "sin_cache tensor must be on CUDA device");
    TORCH_CHECK(cos_cache.dtype() == torch::kFloat16, "cos_cache tensor must be float16");
    TORCH_CHECK(sin_cache.dtype() == torch::kFloat16, "sin_cache tensor must be float16");
    TORCH_CHECK(cos_cache.is_contiguous(), "cos_cache tensor must be contiguous");
    TORCH_CHECK(sin_cache.is_contiguous(), "sin_cache tensor must be contiguous");
    
    // Tensor dimensions for query
    auto q_sizes = q.sizes();
    TORCH_CHECK(q_sizes.size() == 4,
                "Query tensor must be 4D [batch, heads, seq_len, head_dim]");
    int B = q_sizes[0];
    int H = q_sizes[1];
    int N = q_sizes[2];
    int D = q_sizes[3];
    TORCH_CHECK(D % 2 == 0, "head_dim must be even");
    
    // Validate key dimensions match query
    auto k_sizes = k.sizes();
    TORCH_CHECK(k_sizes.size() == 4,
                "Key tensor must be 4D [batch, heads, seq_len, head_dim]");
    TORCH_CHECK(k_sizes[0] == B, "Key batch size must match query");
    TORCH_CHECK(k_sizes[1] == H, "Key num_heads must match query");
    TORCH_CHECK(k_sizes[2] == N, "Key seq_len must match query");
    TORCH_CHECK(k_sizes[3] == D, "Key head_dim must match query");
    
    // Validate cos/sin cache dimensions
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
    
    // Allocate output tensors
    auto q_output = torch::zeros_like(q);
    auto k_output = torch::zeros_like(k);
    
    // Get data pointers
    half *q_ptr = reinterpret_cast<half *>(q.data_ptr<at::Half>());
    half *k_ptr = reinterpret_cast<half *>(k.data_ptr<at::Half>());
    half *q_out_ptr = reinterpret_cast<half *>(q_output.data_ptr<at::Half>());
    half *k_out_ptr = reinterpret_cast<half *>(k_output.data_ptr<at::Half>());
    half *cos_ptr = reinterpret_cast<half *>(cos_cache.data_ptr<at::Half>());
    half *sin_ptr = reinterpret_cast<half *>(sin_cache.data_ptr<at::Half>());
    
    // Launch dual kernel
    rope_apply_half2_dual(q_ptr, k_ptr, q_out_ptr, k_out_ptr, 
                          cos_ptr, sin_ptr, B, H, N, D);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel failed: ", cudaGetErrorString(err));
    
    // Return both outputs as a vector
    return {q_output, k_output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkrope_apply_cuda", &rope_apply_cuda_dual, 
          "RoPE apply CUDA kernel for Q and K",
          py::arg("q"), 
          py::arg("k"), 
          py::arg("cos_cache"), 
          py::arg("sin_cache"));
}