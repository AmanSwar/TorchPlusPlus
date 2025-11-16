"""
Dia Model Architecture and Benchmark Script

This script consolidates the core Dia model architecture and provides
benchmarking utilities to measure inference performance (tokens/sec).
"""

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import RMSNorm


from torchpp.llmops.kvcache import KVCache
from torchpp.dlops.linear import LinearSILU , Dense
from torchpp.attention.gqa import GroupedQueryAttention as SelfAttention
from torchpp.dlops.normalization import RmsNormFused as RMSNorm
# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class EncoderConfig:
    hidden_size: int = 512  # Reduced from 1024
    intermediate_size: int = 2048  # Reduced from 4096
    num_hidden_layers: int = 6  # Reduced from 12
    num_attention_heads: int = 8  # Reduced from 16
    num_key_value_heads: int = 8  # Reduced from 16
    head_dim: int = 64  # Reduced from 128
    max_position_embeddings: int = 512  # Reduced from 1024
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    vocab_size: int = 256


@dataclass
class DecoderConfig:
    hidden_size: int = 512  # Reduced from 2048
    intermediate_size: int = 2048  # Reduced from 8192
    num_hidden_layers: int = 6  # Reduced from 18
    num_attention_heads: int = 8  # Reduced from 16
    num_key_value_heads: int = 2  # Reduced from 4
    head_dim: int = 64  # Reduced from 128
    cross_num_attention_heads: int = 8  # Reduced from 16
    cross_num_key_value_heads: int = 8  # Reduced from 16
    cross_head_dim: int = 64  # Reduced from 128
    max_position_embeddings: int = 512  # Reduced from 3072
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    vocab_size: int = 1028
    num_channels: int = 9


@dataclass
class DiaConfig:
    encoder_config: EncoderConfig
    decoder_config: DecoderConfig
    bos_token_id: int = 1026
    eos_token_id: int = 1024
    pad_token_id: int = 1025
    delay_pattern: list = None

    def __post_init__(self):
        if self.delay_pattern is None:
            self.delay_pattern = [0, 8, 9, 10, 11, 12, 13, 14, 15]


# ============================================================================
# State Management Classes
# ============================================================================

# class KVCache(nn.Module):
#     def __init__(self, batch_size: int, num_heads: int, max_len: int, 
#                  head_dim: int, dtype: torch.dtype, device: torch.device):
#         super().__init__()
#         self.batch_size = batch_size
#         self.num_heads = num_heads
#         self.max_len = max_len
#         self.head_dim = head_dim
#         self.dtype = dtype
#         self.device = device
        
#         # Start with empty cache - will allocate on first use
#         self.k = None
#         self.v = None
#         self.current_len = 0

#     def _ensure_allocated(self, length: int):
#         """Allocate cache only when needed, with the actual length required."""
#         if self.k is None:
#             # Allocate only what we need, not the full max_len
#             alloc_len = min(length, self.max_len)
#             self.k = torch.zeros(
#                 (self.batch_size, self.num_heads, alloc_len, self.head_dim), 
#                 dtype=self.dtype, device=self.device
#             )
#             self.v = torch.zeros(
#                 (self.batch_size, self.num_heads, alloc_len, self.head_dim), 
#                 dtype=self.dtype, device=self.device
#             )

#     def update(self, k: torch.Tensor, v: torch.Tensor, idx: int):
#         self._ensure_allocated(idx + 1)
        
#         # Expand cache if needed
#         if idx >= self.k.shape[2]:
#             new_size = min(idx + 100, self.max_len)  # Grow in chunks
#             new_k = torch.zeros(
#                 (self.batch_size, self.num_heads, new_size, self.head_dim),
#                 dtype=self.dtype, device=self.device
#             )
#             new_v = torch.zeros(
#                 (self.batch_size, self.num_heads, new_size, self.head_dim),
#                 dtype=self.dtype, device=self.device
#             )
#             new_k[:, :, :self.k.shape[2], :] = self.k
#             new_v[:, :, :self.v.shape[2], :] = self.v
#             self.k = new_k
#             self.v = new_v
        
#         self.k[:, :, idx, :] = k.squeeze(2)
#         self.v[:, :, idx, :] = v.squeeze(2)
#         self.current_len = max(self.current_len, idx + 1)
#         return self.k[:, :, :self.current_len, :], self.v[:, :, :self.current_len, :]


@dataclass
class EncoderState:
    positions: torch.Tensor
    attn_mask: torch.Tensor


@dataclass
class DecoderState:
    positions: torch.Tensor
    self_attn_cache: list[KVCache]
    cross_attn_cache: list[KVCache]
    causal_mask: torch.Tensor
    cross_attn_mask: torch.Tensor


# ============================================================================
# Core Model Components
# ============================================================================




class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, embedding_dims: int, max_timescale: float = 10000.0,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if embedding_dims % 2 != 0:
            raise ValueError("Embedding dim must be even for RoPE.")
        
        self.embedding_dims = embedding_dims
        self.compute_dtype = dtype
        
        half_dim = embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half_dim)) / embedding_dims
        timescale = (1.0 * (max_timescale / 1.0) ** fraction).to(torch.float32)
        self.register_buffer("timescale", timescale, persistent=False)

    def get_cos_sin(self, inputs: torch.Tensor, positions: torch.Tensor):
        # inputs: (B, T, H) where T is sequence length
        # positions: (B, T) or (B, max_pos) - we need to slice to match T
        seq_len = inputs.shape[1]
        
        # Slice positions to match input sequence length
        pos = positions[:, :seq_len] if positions.shape[1] > seq_len else positions
        
        # Reshape for broadcasting: (B, T, 1, 1)
        position = pos.unsqueeze(-1).unsqueeze(-1)
        sinusoid_inp = position / self.timescale
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        return cos.to(torch.float16).contiguous() , sin.to(torch.float16).contiguous()
        
       


class MlpBlock(nn.Module):
    """MLP block with gated activation."""
    
    def __init__(self, embed_dim: int, intermediate_dim: int, 
                 compute_dtype: torch.dtype):
        super().__init__()
        self.dtype = compute_dtype 

        self.linear_silu = LinearSILU(embed_dim , intermediate_dim)

        self.wi = nn.Linear(
            in_features=embed_dim,
            out_features=intermediate_dim, 
        )

        self.wo = nn.Linear(
            in_features=intermediate_dim,
            out_features=embed_dim,
        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fused_x = self.wi_fused(x)
        x1 = self.linear_silu(x)
        x2 = self.wi(x)
        hidden = torch.mul(x1 , x2).to(self.dtype)
        return self.wo(hidden)


# class SelfAttention(nn.Module):
#     """Self-attention with RoPE and KV caching."""
    
#     def __init__(self, config, embed_dim: int, num_heads: int, 
#                  num_kv_heads: int, head_dim: int, compute_dtype: torch.dtype):
#         super().__init__()
#         self.num_heads = num_heads
#         self.num_kv_heads = num_kv_heads
#         self.head_dim = head_dim
#         self.num_gqa_groups = num_heads // num_kv_heads
        
#         self.q_proj = Dense(
#             in_featrues=(embed_dim,), out_features=(num_heads, head_dim),
#             axis=(-1,), weight_dtype=compute_dtype
#         )
#         self.k_proj = Dense(
#             in_featrues=(embed_dim,), out_features=(num_kv_heads, head_dim),
#             axis=(-1,), weight_dtype=compute_dtype
#         )
#         self.v_proj = Dense(
#             in_featrues=(embed_dim,), out_features=(num_kv_heads, head_dim),
#             axis=(-1,), weight_dtype=compute_dtype
#         )
#         self.o_proj = Dense(
#             in_featrues=(num_heads, head_dim), out_features=(embed_dim,),
#             axis=(-2, -1), weight_dtype=compute_dtype
#         )
        
#         self.rotary_emb = RotaryEmbedding(
#             head_dim, max_timescale=config.rope_theta, dtype=compute_dtype
#         )

#     def forward(self, x: torch.Tensor, positions: torch.Tensor,
#                 attn_mask: torch.Tensor = None, cache: KVCache = None,
#                 is_causal: bool = False, current_idx: int = None):
        
#         q = self.q_proj(x)
#         k = self.k_proj(x)
#         v = self.v_proj(x)
        
#         q = self.rotary_emb.apply_rope(q, positions)
#         k = self.rotary_emb.apply_rope(k, positions)
        
#         q = q.transpose(1, 2)  # (B, N, T, H)
#         k = k.transpose(1, 2)  # (B, K, T, H)
#         v = v.transpose(1, 2)  # (B, K, T, H)
        
#         if cache is not None and current_idx is not None:
#             k, v = cache.update(k, v, current_idx)
        
#         # Only apply masking up to current length
#         effective_mask = attn_mask
#         if cache is not None and attn_mask is not None:
#             seq_len = k.shape[2]
#             effective_mask = attn_mask[:, :, :, :seq_len] if attn_mask.shape[-1] > seq_len else attn_mask
        
#         attn_out = F.scaled_dot_product_attention(
#             q, k, v, attn_mask=effective_mask, scale=1.0,
#             enable_gqa=(self.num_gqa_groups > 1), is_causal=is_causal
#         )
        
#         attn_out = attn_out.transpose(1, 2).contiguous()
#         return self.o_proj(attn_out)


class CrossAttention(nn.Module):
    """Cross-attention for encoder-decoder."""
    
    def __init__(self, q_dim: int, kv_dim: int, num_q_heads: int,
                 num_kv_heads: int, head_dim: int, compute_dtype: torch.dtype):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_gqa_groups = num_q_heads // num_kv_heads
        
        self.q_proj = Dense(
            in_featrues=(q_dim,), out_features=(num_q_heads, head_dim),
            axis=(-1,), weight_dtype=compute_dtype
        )
        self.k_proj = Dense(
            in_featrues=(kv_dim,), out_features=(num_kv_heads, head_dim),
            axis=(-1,), weight_dtype=compute_dtype
        )
        self.v_proj = Dense(
            in_featrues=(kv_dim,), out_features=(num_kv_heads, head_dim),
            axis=(-1,), weight_dtype=compute_dtype
        )
        self.o_proj = Dense(
            in_featrues=(num_q_heads, head_dim), out_features=(q_dim,),
            axis=(-2, -1), weight_dtype=compute_dtype
        )

    def forward(self, q: torch.Tensor, attn_mask: torch.Tensor = None,
                cache: KVCache = None):
        q_proj = self.q_proj(q).transpose(1, 2)
        
        # Handle cache that might not be fully populated
        k = cache.k_cache if cache.k_cache is not None else cache.k_cache
        v = cache.v_cache if cache.v_cache is not None else cache.v_cache
        
        # Adjust mask to actual cache size
        effective_mask = attn_mask
        if cache is not None and attn_mask is not None and cache.cache_len > 0:
            seq_len = cache.cache_len
            effective_mask = attn_mask[:, :, :, :seq_len] if attn_mask.shape[-1] > seq_len else attn_mask
        
        attn_out = F.scaled_dot_product_attention(
            q_proj, k, v, attn_mask=effective_mask, scale=1.0,
            enable_gqa=(self.num_gqa_groups > 1)
        )
        
        attn_out = attn_out.transpose(1, 2).contiguous()
        return self.o_proj(attn_out)


# ============================================================================
# Encoder and Decoder Layers
# ============================================================================

class EncoderLayer(nn.Module):
    """Transformer encoder layer."""
    
    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        enc_cfg = config.encoder_config
        dim = enc_cfg.hidden_size
        
        self.pre_sa_norm = RMSNorm(dim, eps=enc_cfg.norm_eps)
        self.rope = RotaryEmbedding(config.encoder_config.head_dim,max_timescale=enc_cfg.rope_theta, dtype=compute_dtype)
        self.self_attn = SelfAttention(
            d_in=dim,
            num_heads=enc_cfg.num_attention_heads,
            n_kv_heads=enc_cfg.num_key_value_heads,
            head_dim=enc_cfg.head_dim,
            qk_norm=False,

        )
        self.post_sa_norm = RMSNorm(dim, eps=enc_cfg.norm_eps)
        self.mlp = MlpBlock(dim, enc_cfg.intermediate_size, compute_dtype)
        self.compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor, state: EncoderState):
        residual = x
        cos , sin = self.rope.get_cos_sin(x, state.positions)
        print(cos.shape)
        print(x.dtype)
        print(self.compute_dtype)
        x = self.self_attn(
            self.pre_sa_norm(x),
            cos, sin
        )
        x = residual + x
        
        residual = x
        x = self.mlp(self.post_sa_norm(x).to(self.compute_dtype))
        return residual + x


class DecoderLayer(nn.Module):
    """Transformer decoder layer with cross-attention."""
    
    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        dec_cfg = config.decoder_config
        enc_cfg = config.encoder_config
        dim = dec_cfg.hidden_size
        
        self.pre_sa_norm = RMSNorm(dim, eps=dec_cfg.norm_eps)
        
        self.rope = RotaryEmbedding(config.encoder_config.head_dim,max_timescale=enc_cfg.rope_theta, dtype=compute_dtype)
        self.self_attn = SelfAttention(
            d_in=dim,
            num_heads=dec_cfg.num_attention_heads,
            n_kv_heads=dec_cfg.num_key_value_heads,
            head_dim=dec_cfg.head_dim,
            qk_norm=False,
        )
        
        self.pre_ca_norm = RMSNorm(dim, eps=dec_cfg.norm_eps)
        self.cross_attn = CrossAttention(
            dim, enc_cfg.hidden_size, dec_cfg.cross_num_attention_heads,
            dec_cfg.cross_num_key_value_heads, dec_cfg.cross_head_dim, compute_dtype
        )
        
        self.pre_mlp_norm = RMSNorm(dim, eps=dec_cfg.norm_eps)
        self.mlp = MlpBlock(dim, dec_cfg.intermediate_size, compute_dtype)
        self.compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor, state: DecoderState, 
                self_cache: KVCache, cross_cache: KVCache, current_idx: int):
        residual = x
        cos , sin = self.rope.get_cos_sin(x, state.positions)
        x = self.self_attn(
            self.pre_sa_norm(x).to(self.compute_dtype),
            cos, sin,
        )
        x = residual + x
        
        residual = x
        x = self.cross_attn(
            self.pre_ca_norm(x).to(self.compute_dtype),
            attn_mask=state.cross_attn_mask, cache=cross_cache
        )
        x = residual + x
        
        residual = x
        x = self.mlp(self.pre_mlp_norm(x).to(self.compute_dtype))
        return residual + x


# ============================================================================
# Full Encoder and Decoder
# ============================================================================

class Encoder(nn.Module):
    """Full transformer encoder."""
    
    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        enc_cfg = config.encoder_config
        
        self.embedding = nn.Embedding(
            enc_cfg.vocab_size, enc_cfg.hidden_size, dtype=compute_dtype
        )
        self.layers = nn.ModuleList([
            EncoderLayer(config, compute_dtype) 
            for _ in range(enc_cfg.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            enc_cfg.hidden_size, eps=enc_cfg.norm_eps
        )
        self.compute_dtype = compute_dtype

    def forward(self, x_ids: torch.Tensor, state: EncoderState):
        x = self.embedding(x_ids).to(torch.float16)
        for layer in self.layers:
            x = layer(x, state)
        return self.norm(x).to(self.compute_dtype)


class Decoder(nn.Module):
    """Full transformer decoder with cross-attention."""
    
    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        dec_cfg = config.decoder_config
        self.num_channels = dec_cfg.num_channels
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(dec_cfg.vocab_size, dec_cfg.hidden_size, dtype=compute_dtype)
            for _ in range(self.num_channels)
        ])
        
        self.layers = nn.ModuleList([
            DecoderLayer(config, compute_dtype)
            for _ in range(dec_cfg.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(
            dec_cfg.hidden_size, eps=dec_cfg.norm_eps
        )
        
        self.logits_proj = Dense(
            in_featrues=(dec_cfg.hidden_size,),
            out_features=(self.num_channels, dec_cfg.vocab_size),
            axis=(-1,), weight_dtype=compute_dtype
        )
        self.compute_dtype = compute_dtype

    def precompute_cross_attn_cache(self, enc_out: torch.Tensor,
                                     config: DiaConfig) -> list[KVCache]:
        """Precompute cross-attention K/V for all layers."""
        caches = []
        B, S, _ = enc_out.shape
        
        for layer in self.layers:
            k = layer.cross_attn.k_proj(enc_out).transpose(1, 2)
            v = layer.cross_attn.v_proj(enc_out).transpose(1, 2)
            
            cache = KVCache(
                k.shape[0], k.shape[1], S, k.shape[3],  # Use actual seq len, not max
                k.dtype, k.device
            )
            cache._ensure_allocated(S)
            cache.k_cache = k
            cache.v_cache = v
            cache.cache_len = S
            caches.append(cache)
        return caches

    def forward(self, tokens: torch.Tensor, state: DecoderState, 
                current_idx: int):
        # Sum embeddings across channels
        x = sum(self.embeddings[i](tokens[..., i]) 
                for i in range(self.num_channels))
        
        for i, layer in enumerate(self.layers):
            x = layer(x, state, state.self_attn_cache[i],
                     state.cross_attn_cache[i], current_idx)
        
        x = self.norm(x).to(self.compute_dtype)
        logits = self.logits_proj(x)
        return logits.to(torch.float32)


# ============================================================================
# Complete Dia Model
# ============================================================================

class DiaModel(nn.Module):
    """Complete Dia text-to-speech model."""
    
    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config, compute_dtype)
        self.decoder = Decoder(config, compute_dtype)
        self.compute_dtype = compute_dtype

    def forward(self, text_ids: torch.Tensor, audio_tokens: torch.Tensor,
                enc_state: EncoderState, dec_state: DecoderState, 
                current_idx: int = 0):
        """
        Forward pass through encoder and decoder.
        
        Args:
            text_ids: (B, T_text) - Text token IDs
            audio_tokens: (B, 1, C) - Audio tokens for current step
            enc_state: Encoder state with positions and masks
            dec_state: Decoder state with caches and masks
            current_idx: Current decoding step index
            
        Returns:
            logits: (B, 1, C, V) - Logits for next audio tokens
        """
        enc_out = self.encoder(text_ids, enc_state)
        
        # Precompute cross-attention on first call (check if cache is empty)
        if dec_state.cross_attn_cache[0].k_cache is None:
            dec_state.cross_attn_cache = self.decoder.precompute_cross_attn_cache(
                enc_out, self.config
            )
        
        logits = self.decoder(audio_tokens, dec_state, current_idx)
        return logits


# ============================================================================
# Benchmarking Functions
# ============================================================================

def create_dummy_inputs(config: DiaConfig, batch_size: int = 1, 
                       device: str = "cuda"):
    """Create random dummy inputs for benchmarking."""
    text_len = 64  # Reduced from 128
    text_ids = torch.randint(
        0, config.encoder_config.vocab_size, 
        (batch_size, text_len), device=device
    )
    
    audio_tokens = torch.randint(
        0, config.decoder_config.vocab_size,
        (batch_size, 1, config.decoder_config.num_channels), 
        device=device
    )
    
    return text_ids, audio_tokens


def create_states(config: DiaConfig, batch_size: int, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
    """Initialize encoder and decoder states."""
    
    # Encoder state - use actual input length for positions
    max_text_len = 64  # Match the input size we create
    enc_positions = torch.arange(
        max_text_len,
        dtype=torch.float32, device=device
    ).unsqueeze(0).expand(batch_size, -1)
    
    enc_mask = torch.ones(
        (batch_size, 1, max_text_len, max_text_len),
        dtype=torch.bool, device=device
    )
    
    enc_state = EncoderState(
        positions=enc_positions,
        attn_mask=enc_mask
    )
    
    # Decoder state
    dec_positions = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    
    max_len = config.decoder_config.max_position_embeddings
    causal_mask = torch.tril(
        torch.ones(max_len, max_len, dtype=torch.bool, device=device)
    )
    
    cross_mask = torch.ones(
        (batch_size, 1, 1, max_text_len),  # Use actual text length
        dtype=torch.bool, device=device
    )
    
    # Create KV caches
    self_caches = [
        KVCache(
            max_len, config.decoder_config.num_key_value_heads,
            config.decoder_config.head_dim, dtype
        )
        for _ in range(config.decoder_config.num_hidden_layers)
    ]
    
    cross_caches = [
        KVCache(
            max_text_len, config.decoder_config.cross_num_key_value_heads,
            config.decoder_config.cross_head_dim, dtype
        )
        for _ in range(config.decoder_config.num_hidden_layers)
    ]
    
    dec_state = DecoderState(
        positions=dec_positions,
        self_attn_cache=self_caches,
        cross_attn_cache=cross_caches,
        causal_mask=causal_mask,
        cross_attn_mask=cross_mask
    )
    
    return enc_state, dec_state


def benchmark_model(model: DiaModel, config: DiaConfig, 
                   num_steps: int = 100, batch_size: int = 1,
                   warmup_steps: int = 10, device: str = "cuda"):
    """
    Benchmark the model's inference performance.
    
    Args:
        model: The Dia model to benchmark
        config: Model configuration
        num_steps: Number of decoding steps to simulate
        batch_size: Batch size for inference
        warmup_steps: Number of warmup iterations
        device: Device to run on
        
    Returns:
        dict with performance metrics
    """
    model.eval()
    model.to(device)
    
    text_ids, audio_tokens = create_dummy_inputs(config, batch_size, device)
    enc_state, dec_state = create_states(
        config, batch_size, device, model.compute_dtype
    )
    
    print(f"Benchmarking Dia Model")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Dtype: {model.compute_dtype}")
    print(f"  Steps: {num_steps} (+ {warmup_steps} warmup)")
    
    # Calculate memory usage
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    print("-" * 60)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for i in range(warmup_steps):
            dec_state.positions = torch.tensor([[i]], device=device, dtype=torch.long)
            _ = model(text_ids, audio_tokens, enc_state, dec_state, i)
    
    if device == "cuda":
        warmup_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak memory after warmup: {warmup_mem:.2f} GB")
    
    # Actual benchmark
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    print(f"Running {num_steps} steps...")
    with torch.no_grad():
        for i in range(num_steps):
            dec_state.positions = torch.tensor([[i]], device=device, dtype=torch.long)
            logits = model(text_ids, audio_tokens, enc_state, dec_state, i)
            
            # Sample next token (simplified)
            next_token = torch.argmax(logits, dim=-1)
            audio_tokens = next_token
            
            if (i + 1) % 20 == 0:
                print(f"  Step {i+1}/{num_steps}")
    
    torch.cuda.synchronize() if device == "cuda" else None
    end_time = time.time()
    
    duration = end_time - start_time
    total_tokens = num_steps * batch_size * config.decoder_config.num_channels
    tokens_per_sec = total_tokens / duration
    realtime_factor = tokens_per_sec / (config.decoder_config.num_channels * 86)
    
    results = {
        "duration_sec": duration,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "steps_per_sec": num_steps / duration,
        "realtime_factor": realtime_factor,
        "batch_size": batch_size,
        "num_channels": config.decoder_config.num_channels,
    }
    
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        results["peak_memory_gb"] = peak_mem
    
    print(f"\nResults:")
    print(f"  Duration: {duration:.3f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Tokens/sec: {tokens_per_sec:.2f}")
    print(f"  Steps/sec: {results['steps_per_sec']:.2f}")
    print(f"  Realtime factor: {realtime_factor:.2f}x")
    if device == "cuda":
        print(f"  Peak memory: {peak_mem:.2f} GB")
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main function to demonstrate model architecture and benchmark."""
    
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark Dia Model')
    parser.add_argument('--size', type=str, default='tiny', 
                       choices=['tiny', 'small', 'base', 'full'],
                       help='Model size preset')
    parser.add_argument('--steps', type=int, default=50,
                       help='Number of generation steps')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
    args = parser.parse_args()
    
    # Model size presets
    if args.size == 'tiny':
        # Ultra small for testing (100M params)
        enc_config = EncoderConfig(
            hidden_size=256, intermediate_size=1024, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=4, head_dim=64,
            max_position_embeddings=256
        )
        dec_config = DecoderConfig(
            hidden_size=256, intermediate_size=1024, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=64,
            cross_num_attention_heads=4, cross_num_key_value_heads=4, cross_head_dim=64,
            max_position_embeddings=256
        )
    elif args.size == 'small':
        # Small (300M params)
        enc_config = EncoderConfig(
            hidden_size=512, intermediate_size=2048, num_hidden_layers=6,
            num_attention_heads=8, num_key_value_heads=8, head_dim=64,
            max_position_embeddings=512
        )
        dec_config = DecoderConfig(
            hidden_size=512, intermediate_size=2048, num_hidden_layers=6,
            num_attention_heads=8, num_key_value_heads=2, head_dim=64,
            cross_num_attention_heads=8, cross_num_key_value_heads=8, cross_head_dim=64,
            max_position_embeddings=512
        )
    elif args.size == 'base':
        # Base (800M params) - similar to original defaults but smaller
        enc_config = EncoderConfig(
            hidden_size=768, intermediate_size=3072, num_hidden_layers=8,
            num_attention_heads=12, num_key_value_heads=12, head_dim=64
        )
        dec_config = DecoderConfig(
            hidden_size=1024, intermediate_size=4096, num_hidden_layers=12,
            num_attention_heads=16, num_key_value_heads=4, head_dim=64,
            cross_num_attention_heads=12, cross_num_key_value_heads=12, cross_head_dim=64
        )
    else:  # full
        # Full size (1.6B params) - original config
        enc_config = EncoderConfig()
        dec_config = DecoderConfig()
    
    config = DiaConfig(
        encoder_config=enc_config,
        decoder_config=dec_config
    )
    
    # Determine device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda")
    if device == "cpu":
        print("Warning: CUDA not available, using CPU (will be slow)")
        print("Recommend using --size tiny for CPU")
    
    # Create model
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"\n{'='*60}")
    print(f"Model Configuration: {args.size.upper()}")
    print(f"{'='*60}")
    
    model = DiaModel(config, compute_dtype=dtype).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Architecture:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Encoder layers: {config.encoder_config.num_hidden_layers}")
    print(f"  Decoder layers: {config.decoder_config.num_hidden_layers}")
    print(f"  Encoder dim: {config.encoder_config.hidden_size}")
    print(f"  Decoder dim: {config.decoder_config.hidden_size}")
    print()
    
    # Test forward pass
    print("Testing forward pass...")
    text_ids, audio_tokens = create_dummy_inputs(config, batch_size=args.batch_size, device=device)
    enc_state, dec_state = create_states(config, batch_size=args.batch_size, device=device, dtype=dtype)
    
    with torch.no_grad():
        logits = model(text_ids, audio_tokens, enc_state, dec_state, 0)
        print(f"  Input text shape: {text_ids.shape}")
        print(f"  Input audio shape: {audio_tokens.shape}")
        print(f"  Output logits shape: {logits.shape}")
        print("  ✓ Forward pass successful!\n")
    
    # Run benchmark
    benchmark_model(
        model, config,
        num_steps=args.steps,
        batch_size=args.batch_size,
        warmup_steps=5,
        device=device
    )


if __name__ == "__main__":
    main()