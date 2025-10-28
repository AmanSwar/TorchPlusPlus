#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
  const unsigned int MASK = 0xffffffffu;
#pragma unroll
  for (int offset = (WARP_SIZE >> 1); offset >= 1; offset >>= 1) {
    v += __shfl_xor_sync(MASK, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_reduce_sum_f32(float *sdata, float val) {
  int tid = threadIdx.x;
  int lane = tid % WARP_SIZE;
  int wid = tid / WARP_SIZE; // warp id

  val = warp_reduce_sum_f32(val);

  if (lane == 0)
    sdata[wid] = val;

  __syncthreads();

  float total = 0.0f;
  if (wid == 0) {
    int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    float v = (lane < numWarps) ? sdata[lane] : 0.0f;
    v = warp_reduce_sum_f32(v);
    if (lane == 0)
      sdata[0] = v;
  }

  __syncthreads();
  total = sdata[0];
  return total;
}

__global__ void __launch_bounds__(256, 2)
    rmsnorm_vectorized_kernel_fp16(const half2 *__restrict__ input_matrix_ptr,
                                   const half2 *__restrict__ weight_ptr,
                                   half2 *__restrict__ output_matrix_ptr, int M,
                                   int N, float eps) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  if (row >= M)
    return;

  int vcols = (N + 1) / 2;
  int row_start = row * vcols;

  //shared mem
  extern __shared__ char shared_mem[];

  half2 *smem = reinterpret_cast<half2 *>(shared_mem);
  //partial sum to be stored in float for accuracy
  float *smem_partial_sum = reinterpret_cast<float *>(smem + vcols);

  float partial = 0.0f;

  int full_pairs = N / 2;
  bool has_tail = (N % 2) != 0;

  for (int idx = tid; idx < full_pairs; idx += blockDim.x) {
    half2 element = input_matrix_ptr[row_start + idx];
    float fx = __half2float(element.x);
    float fy = __half2float(element.y);
    partial += fx * fx + fy * fy;
    smem[idx] = element;
  }

  // Handle tail element if N is odd
  if (has_tail) {
    int tail_idx = full_pairs;
    if (tid == 0) {
      // Load the tail element from input
      half tail_element = reinterpret_cast<const half *>(input_matrix_ptr)[row * N + N - 1];
      float fx = __half2float(tail_element);
      partial += fx * fx;

      // Store in shared memory as half2 with zero padding
      half2 new_el;
      new_el.x = tail_element;
      new_el.y = __float2half(0.0f);
      smem[tail_idx] = new_el;
    }
  }

  float total_sum = block_reduce_sum_f32(smem_partial_sum, partial);

  float inv_rms = rsqrtf((total_sum / float(N)) + eps);

  // Apply normalization and scaling
  for (int idx = tid; idx < vcols; idx += blockDim.x) {
    half2 element = smem[idx];
    half2 w =
        weight_ptr[idx]; // Fixed: weights are indexed directly, not per row

    float fx = __half2float(element.x) * inv_rms;
    float fy = __half2float(element.y) * inv_rms;

    float out_x = fx * __half2float(w.x);
    float out_y = fy * __half2float(w.y);

    half2 store;
    store.x = __float2half(out_x);
    store.y = __float2half(out_y);

    output_matrix_ptr[row_start + idx] = store;
  }
}

void launch_rmsnorm_fp16_vectorized(const half *input_matrix,
                                    const half *weight_matrix, half *out_matrix,
                                    int M, int N, float eps = 1e-6f) {
  int threads_per_block = 256;
  int blocks_per_grid = M;

  int vcols = (N + 1) / 2;

  // Calculate shared memory size: space for half2 cache + float reduction array
  size_t cache_size = vcols * sizeof(half2);
  size_t reduction_size = WARP_SIZE * sizeof(float);
  size_t smem_size = cache_size + reduction_size;

  const half2 *input_matrix_2 = reinterpret_cast<const half2 *>(input_matrix);
  const half2 *weight_matrix_2 = reinterpret_cast<const half2 *>(weight_matrix);
  half2 *output_matrix_2 = reinterpret_cast<half2 *>(out_matrix);

  rmsnorm_vectorized_kernel_fp16<<<blocks_per_grid, threads_per_block,
                                   smem_size>>>(input_matrix_2, weight_matrix_2,
                                                output_matrix_2, M, N, eps);
}