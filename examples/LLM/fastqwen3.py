import torch
import torch.nn as nn
from typing import Optional, List

from dataclasses import dataclass

from torchpp.dlops.fused_linear import LinearSILU
from torchpp.dlops.normalization import RmsNormFused
from torchpp.attention.gqa import GroupedQueryAttention
from torchpp.llmops.kvcache import KVCache
from torchpp.llmops.llm import LLMInference



@dataclass
class QwenConfig_float16:
    vocab_size: int = 151936
    context_length: int = 32768
    embed_dim: int = 1024
    n_heads: int = 16
    n_layers: int = 28
    head_dim: int = 128
    n_kv_heads: int = 8
    hidden_dim: int = 3072
    qk_norm: bool = True
    rope_base: float = 1e6
    dtype: torch.dtype = torch.float16





def compute_rope_params(
    head_dim, theta_base, context_length, device="cuda", dtype=torch.float32
):
    assert head_dim % 2 == 0, "head dim must be divisible by 2"
    ar = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inv_freq = 1.0 / (theta_base ** (ar / head_dim))
    pos = torch.arange(context_length, device=device, dtype=dtype)
    angles = pos[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles).to(torch.float16).contiguous()
    sin = torch.sin(angles).to(torch.float16).contiguous()
    return cos, sin





class FFN(nn.Module):
  def __init__(self, in_dim: int, hidden_dim: int):
    super().__init__()

    # self.linear_layer1 = nn.Linear(
    #     in_features=in_dim, out_features=hidden_dim, bias=False, dtype=torch.float16
    # )
    self.fused_linear_layer1 = LinearSILU(in_features=in_dim , out_features=hidden_dim)
    self.linear_layerP = nn.Linear(
        in_features=in_dim, out_features=hidden_dim, bias=False, dtype=torch.float16
    )
    self.silu = nn.SiLU()
    self.linear_layer2 = nn.Linear(
        in_features=hidden_dim, out_features=in_dim, bias=False, dtype=torch.float16
    )

  def forward(self, x):

    # x_l = self.linear_layer1(x)
    # x = self.silu(x_l)

    x = self.fused_linear_layer1(x)

    x_p = self.linear_layerP(x)
    
    x = x * x_p
    
    x = self.linear_layer2(x)
    
    return x

class Transformer(nn.Module):
    def __init__(self, cfg: QwenConfig_float16):
        super().__init__()

        self.attn = GroupedQueryAttention(
            d_in=cfg.embed_dim,
            num_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            n_kv_heads=cfg.n_kv_heads,
            qk_norm=cfg.qk_norm,
            dtype=cfg.dtype,
        )

        self.rms_norm1 = RmsNormFused(cfg.embed_dim, eps=1e-6)
        self.rms_norm2 = RmsNormFused(cfg.embed_dim, eps=1e-6)

        self.ffn = FFN(cfg.embed_dim, cfg.hidden_dim)

    def forward(
        self,
        x,
        cos,
        sin,
        kv_cache: Optional[KVCache] = None,
        cache_position: Optional[int] = None,
    ):
        x_res = x
        x = self.rms_norm1(x)
        x = self.attn(x, cos, sin, kv_cache, cache_position)

        x = x + x_res

        x_res = x
        x = self.rms_norm2(x)
        x = self.ffn(x)
        x = x + x_res

        return x


class FastQwen3(LLMInference):
  
    def __init__(self, cfg: QwenConfig_float16):
        super().__init__()

        self.cfg = cfg

        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype=cfg.dtype)

        self.transformer_blocs = nn.ModuleList(
            [Transformer(cfg=cfg) for _ in range(cfg.n_layers)]
        )

        self.final_rmsnorm = RmsNormFused(cfg.embed_dim)

        self.out_head = nn.Linear(
            cfg.embed_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype
        )

        if cfg.head_dim is None:
            head_dim = cfg.embed_dim // cfg.n_heads
        else:
            head_dim = cfg.head_dim

        self.cos, self.sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg.rope_base,
            context_length=cfg.context_length,
        )

    def get_num_layers(self) -> int:
        return self.cfg.n_layers

    def get_kv_heads(self) -> int:
        return self.cfg.n_kv_heads

    def get_head_dim(self) -> int:
        if self.cfg.head_dim is None:
            return self.cfg.embed_dim // self.cfg.n_heads
        return self.cfg.head_dim

    def get_dtype(self) -> torch.dtype:
        return self.cfg.dtype

    def forward_impl(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[KVCache]] = None,
        cache_position: Optional[int] = None,
    ) -> torch.Tensor:
        N = input_ids.shape[-1]

        if kv_caches is not None and cache_position is not None:
            cos_cache, sin_cache = self.cos, self.sin
        else:
            cos_cache, sin_cache = self.cos[:N], self.sin[:N]

        token_embed: torch.Tensor = self.tok_embed(input_ids)
        x = token_embed

        for i, block in enumerate(self.transformer_blocs):
            if kv_caches is not None:
                x = block(x, cos_cache, sin_cache, kv_caches[i], cache_position)
            else:
                x = block(x, cos_cache, sin_cache)

        x = self.final_rmsnorm(x)
        logits = self.out_head(x)
        return logits



if __name__ == "__main__":
    from fast_qwen.arch.qwen_token import Qwen3Tokenizer

    device = torch.device("cuda")
    config = QwenConfig_float16()
    
    model = FastQwen3(config).to(device)
    
    tokenizer_file_path = "/path/to/tokenizer.json"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        add_gen_prompt=True,
        add_thinking=True,
    )

    prompt = "Write a concise summary of why distributed training matters.\n"
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], device=device)

    print("Method 1: Top-p sampling")
    output = model.generate(
        input_ids,
        max_new_token=128,
        temp=0.7,
        top_p=0.9,
        sampling_strategy="top_p",
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(output[0].tolist()))


    print("\nMethod 2: Greedy decoding")
    output = model.generate(
        input_ids,
        max_new_token=128,
        sampling_strategy="greedy",
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(output[0].tolist()))


    print("\nMethod 3: Top-k sampling")
    output = model.generate(
        input_ids,
        max_new_token=128,
        temp=0.8,
        top_k=50,
        sampling_strategy="top_k",
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(output[0].tolist()))


    print("\nMethod 4: Streaming generation")
    
    def stream_callback(token):
        """Called for each generated token"""
        decoded = tokenizer.decode(token[0].tolist())
        print(decoded, end="", flush=True)
    
    output = model.generate(
        input_ids,
        max_new_token=128,
        temp=0.7,
        top_p=0.9,
        sampling_strategy="top_p",
        stream=True,
        stream_callback=stream_callback,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("\n")

    print("\nCache Info:")
    cache_info = model.get_cache_info()
    print(cache_info)


    print("\nMemory Estimation for 2048 tokens:")
    memory_info = model.estimate_memory_usage(max_seq_len=2048)
    print(f"Total memory: {memory_info['total_mb']:.2f} MB")
    print(f"Per layer: {memory_info['per_layer_mb']:.2f} MB")

   
    print("\nManual cache control:")
    model.setup_kv_cache(max_seq_len=512)
    print(f"Cache initialized: {model.get_cache_info()['initialized']}")
    
    model.reset_kv_cache()
    print(f"Cache length after reset: {model.get_cache_info()['cache_length']}")
    
    model.clear_kv_cache()
    print(f"Cache cleared: {not model.get_cache_info()['initialized']}")