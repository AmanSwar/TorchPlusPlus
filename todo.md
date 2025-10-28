(both backward and forward)

- attention
  - vanilla 
  - grouped query
  - multihead
  - multi latent
  - paged attention
  - ring attention
  - vertical slash attention
  - kv_cache


- activation (including quantized version too)
  - SiLU
  - GeLU
  - GeLU-tanh
  - FATReLU
  - SwiGLU

- gemm
  - gptq gemm
  - marlin optimized gptq
  - nvfp4_scaled_mm_kernels
  - awq gemm kernels
  - allspark qgemm w8a16
  - cutlass w4a8 gemm
  - gguf gemm kernel
  - int 8 gemm kernel

- MoE
  - moe_gemm
  - grouped top k
  - top k softmax
  - moe align and sum 
  - moe permute 
  - marlin moe
  - moe permute unpermute

- Sparse
  - sparse gemm
  - 2:4 sparsity kernel


- DLops
  - RMSNorm
  - LayerNorm
  - softmax
  - fused Linear layers with activation
  - RoPE


- blas
  - HGEMV
  - reduction
    - sum
    - max
  - vector addition
  
