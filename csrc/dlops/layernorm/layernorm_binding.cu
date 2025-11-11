#include <torch/extension.h>
#include <cuda_runtime.h>

#include "layernorm_fp16.cuh"


torch::Tensor layernorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float eps = 1e-6f
) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "Input must be float16");
    TORCH_CHECK(weight.dtype() == torch::kFloat16, "Weight must be float16");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    
    auto sizes = input.sizes();
    TORCH_CHECK(sizes.size() >= 2, "Input must be at least 2D");
    
    int M = 1;
    for (int i = 0; i < sizes.size() - 1; i++) {
        M *= sizes[i];
    }

    int N = sizes[sizes.size() - 1];
    
    TORCH_CHECK(weight.numel() == N,  "Weight size must match last dimension of input");
    
    // Allocate output tensor
    auto output = torch::empty_like(input);
    
    // Get data pointers
    const half* input_ptr = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
    const half* weight_ptr = reinterpret_cast<const half*>(weight.data_ptr<at::Half>());
    half* output_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());
    
    // Launch kernel
    launchLayerNorm(input_ptr, weight_ptr, output_ptr, M, N, eps);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("applyLayernorm", &layernorm_forward, "LayerNorm forward (CUDA)");
}