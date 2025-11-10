#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../utils.cuh"
#include "../math_util.h"

namespace torchpp{


template <
    const int kHeadDim,          // Headdim, 32,64,128
    const int kMmaAtomM,         // MMA Atom M, 16
    const int kMmaAtomN,         // MMA Atom N, 8
    const int kMmaAtomK,         // MMA Atom K, 16
    const int kMmaTileSeqLenQ,   // 4, more MMA(warp), M=16*4=64, Q@K^T=[Br(M),
                                 // d(K)]@[d(K),  Bc(N)]
    const int kMmaTileSeqLenK,   // 1, more MMA(warp), N=8*1 =8,  Q@K^T=[Br(M),
                                 // d(K)]@[d(K),  Bc(N)]
    const int kMmaTileSeqLenP,   // 4, more MMA(warp), M=16*4=64, P@V
                                 // =[Br(M),Bc(K)]@[Bc(K), d(N) ]
    const int kMmaTileHeadDimV,  // 1, more MMA(warp), N=8*1 =8,  P@V
                                 // =[Br(M),Bc(K)]@[Bc(K), d(N) ]
    const int kWarpTileSeqLenQ,  // 1, more values, M, Br=64*1=64, matmul M
    const int kWarpTileSeqLenK,  // 8, more values, N, Bc=8*8 =64, matmul N
    const int kWarpTileSeqLenP,  // 1, more values, M, Br=64*1=64, matmul M
    const int kWarpTileHeadDimV, // 8, more values, N,
                                 // d=8*(1|2|3|4|...)=8|...|32|64|96|128|...
    const int kOStorageAccFloat32, // 0/1, MMA Acc always be fp32, but O
                                   // storage can be fp32 or half.
    const int kStage,              // 1,2
    const int kPadQ,               // Pad Q/K/V 0,8
    const int kPadK, const int kPadV
  >
  __global__ void __launch_bounds__(WARP_SIZE *kMmaTileSeqLenQ *kMmaTileSeqLenK)
  flash_attn_mma_stages_split_q_shared_qkv_acc_f32_rr_kernel(
    half *Q, half *K,
    half *V, half *O,
    int QKV_seqlen,
    int QKV_head
  ){
    // calculate the constants -> Br , Bc , num threads , Tc , scale
    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // 16*4*1 -> 64
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; // 8 * 1 * 8 = 64
    constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 128
    const int Tc = divCeil(QKV_seqlen, Bc); // Tc K_tile[Bc,d]
    const float scale = 1.0f / sqrt((float)kHeadDim);

    // indexing 
    const int QKV_batch_id = blockIdx.y / QKV_head; // Batch size
    const int QKV_head_id = blockIdx.y % QKV_head;  // Head num
    const int Q_tile_id = blockIdx.x; // Q tile_id, range [0, Tr]
    const int O_tile_id = Q_tile_id; // O tile_id, same as Q.
    const int tid = threadIdx.x; // within block
    const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
    const int lane_id = tid % WARP_SIZE; // 0~31
    const int warp_QP = warp_id; // 0,1,2,3 or 0~7
    const int warp_KV = 0; // 0

    // offset for Q , K , V and O
    const int Q_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
       (QKV_head_id * QKV_seqlen * kHeadDim)); // Q [seqlen,d]
    const int K_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
       (QKV_head_id * QKV_seqlen * kHeadDim)); // K [seqlen,d]
    const int V_gmem_offset = Q_gmem_offset;// V [seqlen,d]
    const int O_gmem_offset = Q_gmem_offset;  // O [seqlen,d]
    

    int load_smem_Q_Br = (tid / (kNumThreads / Br)); // Br 64 -> tid / 2, row 0 - 64
    int load_smem_Q_d = (tid % (kNumThreads / Br)) * (kHeadDim / (kNumThreads / Br));

    int load_smem_K_Bc = (tid / (kNumThreads / Bc)); // Bc 64, tid / 2, row 0 -64
    int load_smem_K_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));

    int load_smem_V_Bc = (tid / (kNumThreads / Bc)); // Bc 64, tid / 2, row 0 - 64
    int load_smem_V_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));

    int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;

    if (load_gmem_Q_Br >= QKV_seqlen)
      return;
    // KV tile gmem load index starts from 0 and increments with
    // each iteration as we loop over seqlen.
    int load_gmem_K_Bc_offset = 0;
    int load_gmem_V_Bc_offset = 0;

    extern __shared__ half smem[];

    constexpr int Q_tile_size = Br * (kHeadDim + kPadQ); // 64 * 64 * sizeof(half) = 8192 bytes
    constexpr int K_tile_size = Bc * (kHeadDim + kPadK); // K[Bc,d]
    constexpr int V_tile_size = Bc * (kHeadDim + kPadV); // V[Bc,d]

    half *Q_tile_smem = smem; // 8M/16M
    half *K_tile_smem = Q_tile_smem; // QKV shared the same smem
    half *V_tile_smem = Q_tile_smem; // QKV shared the same smem
    
    //for async copies -> convert generic ptrs to shared 
    uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
    uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
    uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

    float lane_block_row_max_old[kWarpTileSeqLenQ][2]; // [1][2]
    float lane_block_row_sum_old[kWarpTileSeqLenQ][2]; // [1][2]

    fill2DRegs<float, kWarpTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
    fill2DRegs<float, kWarpTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);


    // shared qkv only supports head dim <= 512 for now
    static_assert(kHeadDim <= 512, "shared qkv only supports head dim <= 512");

    constexpr bool kCanPrefetchQs2r = ((kHeadDim / kMmaAtomK) <= 32);
    constexpr bool kCanPrefetchKVg2s = (((Q_tile_size / 
      K_tile_size > V_tile_size ? K_tile_size : V_tile_size)) >= 2) && (kStage >= 2);

    constexpr int kPrefetchKg2sSmemId = 0; // smem id for K g2s, 0.
    constexpr int kPrefetchVg2sSmemId = kCanPrefetchKVg2s ? 1 : 0; // smem id for V g2s, 1.
    constexpr int kNumPrefetchQs2r = (kCanPrefetchQs2r) ? (kHeadDim / kMmaAtomK) : 1;
    
    uint32_t R_Q[kNumPrefetchQs2r][kWarpTileSeqLenQ][4]; // [4/8/1][1][4]
    uint32_t R_K[kWarpTileSeqLenK][2]; // [8][2]
    uint32_t R_V[2]; 

    uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][4]; // [1][8][4], acc f32.
    uint32_t R_O[4];

    uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV][(kOStorageAccFloat32) ? 4 : 2];
    fill3DRegs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, ((kOStorageAccFloat32) ? 4 : 2)>(R_D, 0);
    
    // load Q tile from gmem to smem , only load once
    {
      int load_gmem_Q_d = load_smem_Q_d;
      int load_gmem_Q_addr =
          (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
      uint32_t load_smem_Q_ptr =
          (smem_Q_base_ptr +
          (load_smem_Q_Br * (kHeadDim + kPadQ) + load_smem_Q_d) * sizeof(half));
      #pragma unroll
      for (int i = 0; i < (kHeadDim / (kNumThreads / Br)); i += 8) {
        CP_ASYNC_CG(load_smem_Q_ptr + i * 2, &Q[load_gmem_Q_addr + i], 16);
      }
      CP_ASYNC_COMMIT_GROUP();
    }


    #pragma unroll 1
    for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; ++tile_K_seqlen) {
      static_assert(kCanPrefetchQs2r); // always prefetch Q s2r.
      if constexpr (kCanPrefetchQs2r) {
        // Wait Q ready and let K copy async, then prefetch Q from smem -> regs.
        // NOTE: we only need to load Q once from smem -> regs, and then reuse it.
        if (tile_K_seqlen == 0) {
          CP_ASYNC_WAIT_GROUP(0);
          __syncthreads();
          
          #pragma unroll
          for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
            #pragma unroll
            for (int i = 0; i < kWarpTileSeqLenQ; ++i) { // Q[Br,d]=[M,K]
              int warp_smem_Q_Br =  warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
              int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16; // 0~15
              int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8; // 0,8
              uint32_t lane_smem_Q_ptr = (smem_Q_base_ptr + (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d) * sizeof(half));
              LDMATRIX_X4(R_Q[tile_K_d][i][0], R_Q[tile_K_d][i][1],
                          R_Q[tile_K_d][i][2], R_Q[tile_K_d][i][3],
                          lane_smem_Q_ptr); // now, R_Q[1/2/4/8][1][4]
            }
          }
          __syncthreads();
        }
      } // end if kCanPrefetchQs2r

      // Load K tile from gmem -> smem, always use smem part 0.
    // must after prefetch Q s2r in order to reuse Q smem.
    if constexpr (kCanPrefetchKVg2s) {
      if (tile_K_seqlen == 0) {
        load_gmem_K_Bc_offset = tile_K_seqlen * Bc; // e.g (0~3)*64=(0,64,128,192,...)
        int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
        int load_gmem_K_d = load_smem_K_d;
        int load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
        uint32_t load_smem_K_ptr = (smem_K_base_ptr + (kPrefetchKg2sSmemId * K_tile_size +
              load_smem_K_Bc * (kHeadDim + kPadK) + load_smem_K_d) * sizeof(half));

        #pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

        // Now, we have to wait curr K tile ready for Q@K^T MMA.
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
      }
      // load V tile async from gmem -> smem1 before Q@K
      {
        load_gmem_V_Bc_offset = tile_K_seqlen * Bc; // e.g (0~3)*64=(0,64,128,192,...)
        int load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
        int load_gmem_V_d = load_smem_V_d;
        int load_gmem_V_addr = (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
        
        uint32_t load_smem_V_ptr = (smem_V_base_ptr + (kPrefetchVg2sSmemId * V_tile_size +
              load_smem_V_Bc * (kHeadDim + kPadV) + load_smem_V_d) * sizeof(half));
        
        #pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
        }

        CP_ASYNC_COMMIT_GROUP();
      }

    }
    else{
      load_gmem_K_Bc_offset = tile_K_seqlen * Bc; // e.g (0~3)*64=(0,64,128,192,...)
      int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
      int load_gmem_K_d = load_smem_K_d;
      int load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
      uint32_t load_smem_K_ptr = (smem_K_base_ptr + (kPrefetchKg2sSmemId * K_tile_size +
            load_smem_K_Bc * (kHeadDim + kPadK) + load_smem_K_d) * sizeof(half));
      
      #pragma unroll
      for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
        CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
      }
      
      CP_ASYNC_COMMIT_GROUP();
      // Now, we have to wait curr K tile ready for Q@K^T MMA.
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads();

    }
    fill3DRegs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 4>(R_S, 0);
    
    #pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
    // smem -> reg, load k16n8 from smem K, offset d according tile_K_d.
    // ldmatrix.x2 for K_tile_smem, [Bc,kMmaAtomK] from [Bc,d]=[K,N]
      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) { // 8, 16, 32, ...
        // load k16n8 via ldmatrix.x2 from K_tile_smem[Bc,d].
        // K[Bc,d] with row major means K^T[d,Bc] in col major.
        int warp_smem_K_Bc = warp_KV * (kMmaAtomN * kWarpTileSeqLenK) + j * kMmaAtomN;
        int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8; // 0~7
        int lane_smem_K_d = tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8; // 0,8
        uint32_t lane_smem_K_ptr = (smem_K_base_ptr + (kPrefetchKg2sSmemId * K_tile_size +
              lane_smem_K_Bc * (kHeadDim + kPadK) + lane_smem_K_d) * sizeof(half));
        
              LDMATRIX_X2(R_K[j][0], R_K[j][1], lane_smem_K_ptr); // R_K
      } // end for kWarpTileSeqLenK

      // MMA compute
      static_assert(kWarpTileSeqLenQ == 1);
      { // kWarpTileSeqLenQ = 1
        #pragma unroll
        for (int j = 0; j < kWarpTileSeqLenK; ++j) { // 8, 16, 32, ...
          // MMA always accumulate with F32 dtype for high precision.
          HMMA16816F32(R_S[0][j][0], R_S[0][j][1], R_S[0][j][2], R_S[0][j][3],
                       R_Q[tile_K_d][0][0], R_Q[tile_K_d][0][1],
                       R_Q[tile_K_d][0][2], R_Q[tile_K_d][0][3], R_K[j][0],
                       R_K[j][1], R_S[0][j][0], R_S[0][j][1], R_S[0][j][2],
                       R_S[0][j][3]);
        }
      }
    } // end loop over d, S=Q@K^T
    __syncthreads();
    if constexpr (!kCanPrefetchKVg2s) {
      load_gmem_V_Bc_offset = tile_K_seqlen * Bc; // e.g (0~3)*64=(0,64,128,192,...)
      int load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
      int load_gmem_V_d = load_smem_V_d;
      int load_gmem_V_addr = (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
      
      uint32_t load_smem_V_ptr = (smem_V_base_ptr + (kPrefetchVg2sSmemId * V_tile_size +
            load_smem_V_Bc * (kHeadDim + kPadV) + load_smem_V_d) * sizeof(half));
      
      #pragma unroll
      for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
        CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
      }

      CP_ASYNC_COMMIT_GROUP();
    }
    
    //ONLINE SAFE SOFTMAX
    float lane_row_max_new[kWarpTileSeqLenQ][2]; // [1][2]
    float lane_row_sum_new[kWarpTileSeqLenQ][2]; // [1][2]
    fill2DRegs<float, kWarpTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
    fill2DRegs<float, kWarpTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

    static_assert(kWarpTileSeqLenQ == 1);
    {
// Thread level reduce max across kWarpTileSeqLenK dim, namely Bc.
      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        float *t_fptr_S_0_1 = reinterpret_cast<float *>(&(R_S[0][j][0]));
        
        // This should be the row max after S = (Q @ K^T) / sqrt(d)
        float tmp_max_0 = max(t_fptr_S_0_1[0], t_fptr_S_0_1[1]) * scale;
        float tmp_max_1 = max(t_fptr_S_0_1[2], t_fptr_S_0_1[3]) * scale;
        
        lane_row_max_new[0][0] = max(lane_row_max_new[0][0], tmp_max_0);
        lane_row_max_new[0][1] = max(lane_row_max_new[0][1], tmp_max_1);
      } // end for kWarpTileSeqLenK

      // Warp level reduce max, warp_size = 4
      // Each thread contains the maximum of 2 rows of Br,
      // and only the values of T0, T4, ..., T28 are used.
      lane_row_max_new[0][0] = warpReduceMax<float, 4>(lane_row_max_new[0][0]);
      lane_row_max_new[0][1] = warpReduceMax<float, 4>(lane_row_max_new[0][1]);
    } // end for kWarpTileSeqLenQ

    { // kWarpTileSeqLenQ = 1
      // Use latest global row max without update.
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55;
      float block_row_max_new_0 = lane_row_max_new[0][0];
      // Br 1, row_id, 8~15, 24~31, 40~47, 56~63;
      float block_row_max_new_1 = lane_row_max_new[0][1];

      float block_row_max_old_0 = lane_block_row_max_old[0][0];
      float block_row_max_old_1 = lane_block_row_max_old[0][1];
      // Apply m_new = max(m_old, m_new) here.
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        // R_S[][][4] 4 32bit registers with each contains 1 F32 element.
        // (x,y) 0~7->{c0, c1}, (z,w)->8~15 {c2, c3}
        float *t_fptr_S_0_1 = reinterpret_cast<float *>(&(R_S[0][j][0]));
        half *t_hptr_S_0_1 = reinterpret_cast<half *>(&(R_S[0][j][0]));
        // P = Exp(S - m_new), fmaf(x, y, z) = x * y + z in registers;
        t_fptr_S_0_1[0] = __expf(__fmaf_rn(t_fptr_S_0_1[0], scale, -block_row_max_new_0));
        t_fptr_S_0_1[1] = __expf(__fmaf_rn(t_fptr_S_0_1[1], scale, -block_row_max_new_0));
        t_fptr_S_0_1[2] = __expf(__fmaf_rn(t_fptr_S_0_1[2], scale, -block_row_max_new_1));
        t_fptr_S_0_1[3] = __expf(__fmaf_rn(t_fptr_S_0_1[3], scale, -block_row_max_new_1));
        lane_row_sum_new[0][0] += (t_fptr_S_0_1[0] + t_fptr_S_0_1[1]);
        lane_row_sum_new[0][1] += (t_fptr_S_0_1[2] + t_fptr_S_0_1[3]);
        // Update R_S for P[Br,Bc] = Exp(S-m), point wise.
        // Also convert F32 -> half for P@V MMA, reuse R_S as P.
        t_hptr_S_0_1[0] = __float2half_rn(t_fptr_S_0_1[0]);
        t_hptr_S_0_1[1] = __float2half_rn(t_fptr_S_0_1[1]);
        t_hptr_S_0_1[2] = __float2half_rn(t_fptr_S_0_1[2]);
        t_hptr_S_0_1[3] = __float2half_rn(t_fptr_S_0_1[3]);
      } // end for kWarpTileSeqLenK

      // Warp level reduce sum, warp_size = 4
      lane_row_sum_new[0][0] = warpReduceSumF16AccF32<float, 4>(lane_row_sum_new[0][0]);
      lane_row_sum_new[0][1] = warpReduceSumF16AccF32<float, 4>(lane_row_sum_new[0][1]);
    }
    if constexpr (kCanPrefetchKVg2s) {
      if ((tile_K_seqlen + 1) < Tc) {
        CP_ASYNC_WAIT_GROUP(1); // we have send V & K g2s, wait V and let K async.
      } else {
        CP_ASYNC_WAIT_GROUP(0); // we have only send V g2s.
      }
    } else {
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();
    static_assert(kWarpTileSeqLenP == 1);
    {
      // <Prefetch max/sum values>
      // m = max(m_old, m_new), l = exp(m_old - m) * l_old + l_new (FA2 paper)
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55; Br 1, row_id, 8~15, 24~31,
      // 40~47, 56~63
      float block_row_max_new_0 = lane_row_max_new[0][0];
      float block_row_max_new_1 = lane_row_max_new[0][1];
      float block_row_sum_new_0 = lane_row_sum_new[0][0];
      float block_row_sum_new_1 = lane_row_sum_new[0][1];

      float block_row_max_old_0 = lane_block_row_max_old[0][0];
      float block_row_max_old_1 = lane_block_row_max_old[0][1];
      // NOTE: max(-inf, val) = val.
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);
      // Avoid inf value while using m_old for rescaling O.
      block_row_max_old_0 = (tile_K_seqlen > 0 ? block_row_max_old_0 : block_row_max_new_0);
      block_row_max_old_1 = (tile_K_seqlen > 0 ? block_row_max_old_1 : block_row_max_new_1);
      // rescale factor for O and l, exp(m_old - m) for curr tile [Br,d].
      float rescale_o_factor_0 = __expf(block_row_max_old_0 - block_row_max_new_0);
      float rescale_o_factor_1 = __expf(block_row_max_old_1 - block_row_max_new_1);

// Compute P[Br,Bc]@V[Bc,d] = O[Br,d]
// For R_S[1][8][2], mapping the layout below of P matrix.
// MMA = m16n8k16, Br=16x4=64, Bc=8x8=64, layout: 4 warps
// |   64x64   |      warp_KV 0       |
// | warp_QP 0 | MMA 0 ... MMA 0 (x8) |
// | warp_QP 1 | MMA 1 ... MMA 1 (x8) |
// | warp_QP 2 | MMA 2 ... MMA 2 (x8) |
// | warp_QP 3 | MMA 3 ... MMA 3 (x8) |
// tile_V_Bc = 0, all curr MMAs(0~4) need slice P[:,  0:16], 0, 1; stored in all
// MMAs. tile_V_Bc = 1, all curr MMAs(0~4) need slice P[:, 16:32], 2, 3; stored
// in all MMAs. tile_V_Bc = 2, all curr MMAs(0~4) need slice P[:, 32:48], 4, 5;
// stored in all MMAs. tile_V_Bc = 3, all curr MMAs(0~4) need slice P[:, 48:64],
// 6, 7; stored in all MMAs. <HGEMM in registers>
      #pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8, 16, 32, ...
        // Compute d tile, P[Br,Bc]@V[Bc,16] = O[Br,16]
        fill1DRegs<uint32_t, 4>(R_O, 0); // must clear
        
        
        #pragma unroll
        for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
          // Load k16n8 V from smem -> regs, R_KV, ldmatrix.x2.trans.
          int warp_smem_V_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN; // d, matmaul N
          int lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16; // 0~15; Bc, matmul K
          int lane_smem_V_d = warp_smem_V_d;        // 0
          
          uint32_t lane_smem_V_ptr = (smem_V_base_ptr + (kPrefetchVg2sSmemId * V_tile_size +
                lane_smem_V_Bc * (kHeadDim + kPadV) + lane_smem_V_d) * sizeof(half));
          
          LDMATRIX_X2_T(R_V[0], R_V[1], lane_smem_V_ptr); // R_V
          
          int w = tile_V_Bc * 2; // MMA(Warp) selected, 0, 2, 4, 6
          
          // MMA always accumulate with F32 dtype for high precision.
          HMMA16816F32(R_O[0], R_O[1], R_O[2], R_O[3], R_S[0][w][0],
                       R_S[0][w][1], R_S[0][w + 1][0], R_S[0][w + 1][1], R_V[0],
                       R_V[1], R_O[0], R_O[1], R_O[2], R_O[3]);
        } // end for V Bc.
        // Now, we get [Br,8] slice of [Br,d], each warp(MMA) contains m16n8.
        // 0. Rescale O: Online rescaling O each tile_K_seqlen step, need m_new,
        // m_old. m = max(m_old, m_new), O_new[Br,d] = exp(m_old - m) * O_old +
        // P@V use exp(m_old - m_new), not 1/(m_old - m_new). O_new[Br,d] =
        // exp(m_old - m_new) * O_old + P@V
        float *t_fptr_O_0_1 = reinterpret_cast<float *>(&(R_O[0]));
        if constexpr (kOStorageAccFloat32) {
          // (x,y) 0~7->{c0, c1}, (z,w)->8~15 {c2, c3}
          float *t_fptr_D_0_1 = reinterpret_cast<float *>(&(R_D[0][j][0])); // kWarpTileSeqLenP=1
          t_fptr_D_0_1[0] = __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[0], t_fptr_O_0_1[0]);
          t_fptr_D_0_1[1] = __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[1], t_fptr_O_0_1[1]);
          t_fptr_D_0_1[2] = __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[2], t_fptr_O_0_1[2]);
          t_fptr_D_0_1[3] = __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[3], t_fptr_O_0_1[3]);
        } 
        else {
          half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0])); // kWarpTileSeqLenP=1
          t_hptr_D_0_1[0] = __float2half_rn(__fmaf_rn(rescale_o_factor_0, __half2float(t_hptr_D_0_1[0]), t_fptr_O_0_1[0]));
          t_hptr_D_0_1[1] = __float2half_rn(__fmaf_rn(rescale_o_factor_0, __half2float(t_hptr_D_0_1[1]), t_fptr_O_0_1[1]));
          t_hptr_D_0_1[2] = __float2half_rn( __fmaf_rn(rescale_o_factor_1, __half2float(t_hptr_D_0_1[2]), t_fptr_O_0_1[2]));
          t_hptr_D_0_1[3] = __float2half_rn( __fmaf_rn(rescale_o_factor_1, __half2float(t_hptr_D_0_1[3]), t_fptr_O_0_1[3]));
        }
      } // end for kWarpTileHeadDimV.
      // Now, we can update m, l after O has been scaled.
      // 1. First, update block row sum Exp for each lane which
      // need both m_new and m_old.
      float block_row_sum_old_0 = lane_block_row_sum_old[0][0];
      float block_row_sum_old_1 = lane_block_row_sum_old[0][1];

      // Update l = exp(m_old - m_new) * l_old + row_sum(P).
      lane_block_row_sum_old[0][0] = (__fmaf_rn(rescale_o_factor_0, block_row_sum_old_0, block_row_sum_new_0));
      lane_block_row_sum_old[0][1] = (__fmaf_rn(rescale_o_factor_1, block_row_sum_old_1, block_row_sum_new_1));
      
      // 2. Then, update block row max for each lane.
      lane_block_row_max_old[0][0] = block_row_max_new_0;
      lane_block_row_max_old[0][1] = block_row_max_new_1;
    } // end P@V

    __syncthreads();
    if constexpr (kCanPrefetchKVg2s) {
      if ((tile_K_seqlen + 1) < Tc) {
        // now, we have to wait next K tile ready in smem.
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
      }
    }

  }
  __syncthreads();
  
  static_assert(kWarpTileSeqLenP == 1);
  
  { // kWarpTileSeqLenP = 1
    float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[0][0]);
    float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[0][1]);
    
    #pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8, 16, 32, ...
      // Scaling in registers & convert F32 -> half for O collective store.
      if constexpr (kOStorageAccFloat32) {
        float *t_fptr_D_0_1 = reinterpret_cast<float *>(&(R_D[0][j][0]));
        half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0]));
        t_hptr_D_0_1[0] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[0]);
        t_hptr_D_0_1[1] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[1]);
        t_hptr_D_0_1[2] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[2]);
        t_hptr_D_0_1[3] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[3]);
      } 
      else {
        half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0]));
        t_hptr_D_0_1[0] = __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D_0_1[0]));
        t_hptr_D_0_1[1] = __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D_0_1[1]));
        t_hptr_D_0_1[2] = __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D_0_1[2]));
        t_hptr_D_0_1[3] = __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D_0_1[3]));
      }
    } // end for kWarpTileHeadDimV
  } // end for kWarpTileSeqLenP = 1
  
  static_assert(kWarpTileSeqLenP == 1);
  { // kWarpTileSeqLenP = 1
  
    #pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8
      static_assert(kCanPrefetchQs2r && kNumPrefetchQs2r > 1);
      // reuse R_Q[4/8][1][4] for collective store, reduce registers usage.
      R_Q[0][0][0] = R_D[0][j][0];
      R_Q[1][0][0] = R_D[0][j][1]; // warp_size 4
      
      R_Q[0][0][1] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 1, 4);
      R_Q[0][0][2] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 2, 4);
      R_Q[0][0][3] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 3, 4);
      R_Q[1][0][1] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 1, 4);
      R_Q[1][0][2] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 2, 4);
      R_Q[1][0][3] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 3, 4);
      
      // st.global.v4 128 bits. [Br,d]
      if (lane_id % 4 == 0) {
        // (0/1)*32 + (0/1)*16=(0,16,32,48), + 0~7 -> 0~56
        int store_warp_regs_O_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + 0 * kMmaAtomM;
        int store_lane_gmem_O_Br = O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4; // 0~7
        // (0~3)*16 + (0/1)*8=(0,8,16,24,...,48,56)
        int store_warp_regs_O_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
        int store_lane_gmem_O_d = store_warp_regs_O_d; // (0~3)*16+(0/8)
        int store_gmem_O_addr_0 =(O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim +store_lane_gmem_O_d);
        int store_gmem_O_addr_1 = (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim + store_lane_gmem_O_d);
        
        LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(R_Q[0][0][0]);
        LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(R_Q[1][0][0]);
      }
    } // end for kWarpTileHeadDimV
  }
 } // end for kWarpTileSeqLenP = 1

 template <const int kHeadDim, const int kStage>
void launch_flash_attn(
    half* Q, 
    half* K, 
    half* V, 
    half* O,
    int64_t QKV_batch,
    int64_t QKV_head,
    int64_t QKV_seqlen

  ){
  
    constexpr int kMmaAtomM = 16;
    constexpr int kMmaAtomN = 8;
    constexpr int kMmaAtomK = 16;
    
    constexpr int kMmaTileSeqLenQ = (kHeadDim < 128) ? 8 : 8;
    constexpr int kMmaTileSeqLenK = 1;
    constexpr int kMmaTileSeqLenP = (kHeadDim < 128) ? 8 : 8;
    constexpr int kMmaTileHeadDimV = 1;
    constexpr int kWarpTileSeqLenQ = 1;
    constexpr int kWarpTileSeqLenK = (kHeadDim < 128) ? 8 : 8;
    constexpr int kWarpTileSeqLenP = 1;
    constexpr int kWarpTileHeadDimV = (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)); // 8,16,32,....
    
    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // 16*4*1=64
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; //  8*1*8=64
    constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 32*4*1=128, num threads
    
    constexpr int kPadQ = 8;
    constexpr int kPadK = 8;
    constexpr int kPadV = 8;
    if constexpr (kStage > 1) {
      static_assert(((Br / Bc) >= 2));
    }

    constexpr int kOStorageAccFloat32 = 1;
    constexpr int Q_tile_size = (Br * (kHeadDim + kPadQ));
    constexpr int K_tile_size = (Bc * (kHeadDim + kPadK));
    constexpr int V_tile_size = (Bc * (kHeadDim + kPadV));
    int smem_max_size = std::max(Q_tile_size, std::max(K_tile_size, V_tile_size)) * sizeof(half);
    if constexpr (kStage > 1) { // make sure kStage > 1 work
      smem_max_size = smem_max_size > kStage * std::max(K_tile_size, V_tile_size) * sizeof(half)
              ? smem_max_size
              : kStage * std::max(K_tile_size, V_tile_size) * sizeof(half);
    }

    dim3 grid(divCeil(QKV_seqlen, Br), QKV_batch * QKV_head);
    dim3 block(kNumThreads);
    cudaFuncSetAttribute(
      flash_attn_mma_stages_split_q_shared_qkv_acc_f32_rr_kernel<
          kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
          kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
          kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
          kOStorageAccFloat32, kStage, kPadQ, kPadK, kPadV>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      // kMaxSramPerBlock
      98304);

    flash_attn_mma_stages_split_q_shared_qkv_acc_f32_rr_kernel<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
      kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
      kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
      kOStorageAccFloat32, kStage, kPadQ, kPadK, kPadV>
      <<<grid, block, smem_max_size>>>(
        Q,
        K,
        V,
        O,
        QKV_seqlen, 
        QKV_head
      );

  }
}