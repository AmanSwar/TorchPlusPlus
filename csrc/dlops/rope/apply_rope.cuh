#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void rope_apply_half2_kernel(const half2 *__restrict__ x,
                                        const half2 *__restrict__ cos_cache,
                                        const half2 *__restrict__ sin_cache,
                                        half2 *__restrict__ out, int batch_size,
                                        int num_heads, int seq_len,
                                        int head_dim_half) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * num_heads * seq_len * head_dim_half;
  if (idx >= total)
    return;

  int pair_idx = idx % head_dim_half;
  int seq_idx = (idx / head_dim_half) % seq_len;

  half2 xh = x[idx];

  int head_dim_quarter = head_dim_half / 2;
  half2 xh_pair;
  bool first_half = (pair_idx < head_dim_quarter);

  if (first_half) {
    int paired_idx = idx + head_dim_quarter;
    xh_pair = x[paired_idx];
  } else {
    int paired_idx = idx - head_dim_quarter;
    xh_pair = x[paired_idx];
  }

  int cos_sin_idx = seq_idx * head_dim_half + pair_idx;
  half2 cos_h = cos_cache[cos_sin_idx];
  half2 sin_h = sin_cache[cos_sin_idx];

  half2 rotated;
  if (first_half) {
    rotated = make_half2(__hneg(xh_pair.x), __hneg(xh_pair.y));
  } else {
    rotated = xh_pair;
  }

  __half r1 = __hadd(__hmul(xh.x, cos_h.x), __hmul(rotated.x, sin_h.x));
  __half r2 = __hadd(__hmul(xh.y, cos_h.y), __hmul(rotated.y, sin_h.y));

  out[idx] = make_half2(r1, r2);
}

void rope_apply_half2(half *x, half *out, half *cos, half *sin, int B, int H,
                      int N, int D) {

  const int head_dim_half = D / 2;
  const std::size_t total_half2 = (std::size_t)B * H * N * head_dim_half;
  int block_size = 256;
  int grid_size = (total_half2 + block_size - 1) / block_size;

  const half2 *input = reinterpret_cast<const half2 *>(x);
  const half2 *cos_cache = reinterpret_cast<const half2 *>(cos);
  const half2 *sin_cache = reinterpret_cast<const half2 *>(sin);
  half2 *output = reinterpret_cast<half2 *>(out);

  rope_apply_half2_kernel<<<grid_size, block_size>>>(
      input, cos_cache, sin_cache, output, B, H, N, head_dim_half);
}