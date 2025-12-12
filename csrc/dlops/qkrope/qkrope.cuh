#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>



__global__ void rope_apply_half2_dual_kernel(
    const half2 *__restrict__ q_in,
    const half2 *__restrict__ k_in,
    const half2 *__restrict__ cos_cache,
    const half2 *__restrict__ sin_cache,
    half2 *__restrict__ q_out,
    half2 *__restrict__ k_out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim_half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * seq_len * head_dim_half;
    
    if (idx >= total) return;
    
    int pair_idx = idx % head_dim_half;
    int seq_idx = (idx / head_dim_half) % seq_len;
    
    half2 q_xh = q_in[idx];
    half2 k_xh = k_in[idx];
    
    int head_dim_quarter = head_dim_half / 2;
    half2 q_xh_pair, k_xh_pair;
    
    bool first_half = (pair_idx < head_dim_quarter);
    
    if (first_half) {
        int paired_idx = idx + head_dim_quarter;
        q_xh_pair = q_in[paired_idx];
        k_xh_pair = k_in[paired_idx];
    } else {
        int paired_idx = idx - head_dim_quarter;
        q_xh_pair = q_in[paired_idx];
        k_xh_pair = k_in[paired_idx];
    }
    
    int cos_sin_idx = seq_idx * head_dim_half + pair_idx;
    half2 cos_h = cos_cache[cos_sin_idx];
    half2 sin_h = sin_cache[cos_sin_idx];
    
    half2 q_rotated, k_rotated;
    
    if (first_half) {
        q_rotated = make_half2(__hneg(q_xh_pair.x), __hneg(q_xh_pair.y));
        k_rotated = make_half2(__hneg(k_xh_pair.x), __hneg(k_xh_pair.y));
    } else {
        q_rotated = q_xh_pair;
        k_rotated = k_xh_pair;
    }
    
    __half q_r1 = __hadd(__hmul(q_xh.x, cos_h.x), __hmul(q_rotated.x, sin_h.x));
    __half q_r2 = __hadd(__hmul(q_xh.y, cos_h.y), __hmul(q_rotated.y, sin_h.y));
    q_out[idx] = make_half2(q_r1, q_r2);
    
    __half k_r1 = __hadd(__hmul(k_xh.x, cos_h.x), __hmul(k_rotated.x, sin_h.x));
    __half k_r2 = __hadd(__hmul(k_xh.y, cos_h.y), __hmul(k_rotated.y, sin_h.y));
    k_out[idx] = make_half2(k_r1, k_r2);
}

void rope_apply_half2_dual(
    half *q_in, half *k_in,
    half *q_out, half *k_out,
    half *cos, half *sin,
    int B, int H, int N, int D)
{
    const int head_dim_half = D / 2;
    const std::size_t total_half2 = (std::size_t)B * H * N * head_dim_half;
    
    int block_size = 256;
    int grid_size = (total_half2 + block_size - 1) / block_size;
    
    const half2 *q_input = reinterpret_cast<const half2*>(q_in);
    const half2 *k_input = reinterpret_cast<const half2*>(k_in);
    const half2 *cos_cache = reinterpret_cast<const half2*>(cos);
    const half2 *sin_cache = reinterpret_cast<const half2*>(sin);
    half2 *q_output = reinterpret_cast<half2*>(q_out);
    half2 *k_output = reinterpret_cast<half2*>(k_out);
    
    rope_apply_half2_dual_kernel<<<grid_size, block_size>>>(
        q_input, k_input,
        cos_cache, sin_cache,
        q_output, k_output,
        B, H, N, head_dim_half);
}