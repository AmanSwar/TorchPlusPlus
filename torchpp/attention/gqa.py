import torch
import torch.nn as nn

from flash_attn import flash_attn_func

from torchpp.dlops.rope import rope_apply

from typing import Optional 
from torchpp.llmops.kvcache import KVCache

class GroupedQueryAttention(nn.Module):

    def __init__(
        self,
        d_in: int,
        num_heads: int,
        n_kv_heads: int,
        head_dim: int | None = None,
        qk_norm: bool = True,
        dtype : torch.dtype =torch.float16,

        DEBUG : bool = False,
    ):
        super().__init__()

        assert num_heads % n_kv_heads == 0, "Num heads is not divisible by num kv grps"

        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads
        self.num_kv_grps = num_heads // n_kv_heads

        if head_dim is None:
            assert (
                d_in % num_heads == 0
            ), "in dimension must be divisible by number of heads"
            head_dim = d_in // num_heads

        self.head_dim: int = head_dim
        self.d_out = self.head_dim * self.num_heads #embed dim

        self.Wq = nn.Linear(
            in_features=d_in, out_features=self.d_out, bias=False, dtype=dtype
        )

        self.Wk = nn.Linear(
            in_features=d_in,
            out_features=self.n_kv_heads * self.head_dim,
            bias=False,
            dtype=dtype,
        )
        self.Wv = nn.Linear(
            in_features=d_in,
            out_features=self.n_kv_heads * self.head_dim,
            bias=False,
            dtype=dtype,
        )

        self.out_projection = nn.Linear(
            in_features=self.d_out, out_features=d_in, bias=False, dtype=dtype
        )

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6 ,dtype=dtype)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6 ,dtype=dtype)
        else:
            self.q_norm = self.k_norm = None

        self.DEBUG = DEBUG

    def forward(
        self,
        x,
        cos , sin,
        kv_cache : Optional[KVCache] = None,
        cache_pos : Optional[int] = None,
    ):
        if(self.DEBUG):
            print(f"\nGQA Input x shape: {x.shape}") 
        
        bs, seq_len, _ = x.shape

        Q: torch.Tensor = self.Wq(x)
        K: torch.Tensor = self.Wk(x)
        V: torch.Tensor = self.Wv(x)

        Q = Q.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm:
            Q = self.q_norm(Q)

        if self.k_norm:
            K = self.k_norm(K)

        Q = Q.contiguous()
        K = K.contiguous()

        if(self.DEBUG):
            print("\nGQA DEBUG INFO:")

            print(f"Q dtype: {Q.dtype}, K dtype: {K.dtype}, V dtype: {V.dtype}")
            print(f"cos dtype: {cos.dtype}, sin dtype: {sin.dtype}")
            print(f"Q Shape: {Q.shape}, K Shape: {K.shape}, V Shape: {V.shape}")
            print(f"cos Shape: {cos.shape}, sin Shape: {sin.shape}")
            print(f"Q contiguous: {Q.is_contiguous()}, K contiguous: {K.is_contiguous()}")

        if cache_pos is not None and kv_cache is not None:
            pos_cos = cos[cache_pos : cache_pos + seq_len]
            pos_sin = sin[cache_pos : cache_pos + seq_len]
            Q = rope_apply(Q, pos_cos, pos_sin)
            K = rope_apply(K, pos_cos, pos_sin)
        else:
            Q = rope_apply(Q, cos, sin)
            K = rope_apply(K, cos, sin)

        if kv_cache is not None:
            K, V = kv_cache.update(K, V)

        K = K.repeat_interleave(self.num_kv_grps, dim=1)
        V = V.repeat_interleave(self.num_kv_grps, dim=1)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)


        attn_out : torch.Tensor | None = flash_attn_func(Q, K, V, causal=True)

        attn_out = attn_out.reshape(bs, seq_len, self.d_out)
        return self.out_projection(attn_out)
    
    