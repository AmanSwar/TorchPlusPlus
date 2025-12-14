import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
from dataclasses import dataclass

# --- Configuration ---

class EncoderConfigReduced:
    def __init__(self):
        self.model_type = "dia_encoder"
        self.hidden_size = 128  # Reduced
        self.intermediate_size = 512  # Reduced
        self.num_hidden_layers = 2  # Reduced
        self.num_attention_heads = 4  # Reduced
        self.num_key_value_heads = 4  # Reduced
        self.head_dim = 32  # Reduced
        self.hidden_act = "silu"
        self.max_position_embeddings = 1024
        self.initializer_range = 0.02
        self.norm_eps = 1e-5
        self.rope_theta = 10000.0
        self.rope_scaling = None
        self.vocab_size = 256

class EncoderConfig:
    def __init__(self):
        self.model_type = "dia_encoder"
        self.hidden_size = 1024
        self.intermediate_size = 4096
        self.num_hidden_layers = 8
        self.num_attention_heads = 16
        self.num_key_value_heads = 16
        self.head_dim = 128
        self.hidden_act = "silu"
        self.max_position_embeddings = 1024
        self.initializer_range = 0.02
        self.norm_eps = 1e-5
        self.rope_theta = 10000.0
        self.rope_scaling = None
        self.vocab_size = 256

class DecoderConfig:
    def __init__(self):
        self.model_type = "dia_decoder"
        self.hidden_size = 2048
        self.intermediate_size = 8192
        self.num_hidden_layers = 12
        self.num_attention_heads = 16
        self.num_key_value_heads = 4
        self.head_dim = 128
        self.cross_hidden_size = 1024
        self.cross_num_attention_heads = 16
        self.cross_num_key_value_heads = 16
        self.cross_head_dim = 128
        self.hidden_act = "silu"
        self.max_position_embeddings = 3072
        self.initializer_range = 0.02
        self.norm_eps = 1e-5
        self.rope_theta = 10000.0
        self.rope_scaling = None
        self.vocab_size = 1028
        self.num_channels = 9


class DecoderConfigReduced:
    def __init__(self):
        self.model_type = "dia_decoder"
        self.hidden_size = 256  # Reduced
        self.intermediate_size = 1024  # Reduced
        self.num_hidden_layers = 2  # Reduced
        self.num_attention_heads = 4  # Reduced
        self.num_key_value_heads = 2  # Reduced
        self.head_dim = 64  # Reduced
        self.cross_hidden_size = 128  # Reduced, matches new Encoder hidden_size
        self.cross_num_attention_heads = 4  # Reduced
        self.cross_num_key_value_heads = 4  # Reduced
        self.cross_head_dim = 32  # Reduced
        self.hidden_act = "silu"
        self.max_position_embeddings = 3072
        self.initializer_range = 0.02
        self.norm_eps = 1e-5
        self.rope_theta = 10000.0
        self.rope_scaling = None
        self.vocab_size = 1028
        self.num_channels = 9

class DiaConfig:
    def __init__(self):
        self.model_type = "dia"
        self.is_encoder_decoder = True
        self.encoder_config = EncoderConfig()
        self.decoder_config = DecoderConfig()
        self.src_vocab_size = self.encoder_config.vocab_size
        self.tgt_vocab_size = self.decoder_config.vocab_size
        self.initializer_range = 0.02
        self.norm_eps = 1e-5
        self.torch_dtype = "float32"
        self.bos_token_id = 1026
        self.eos_token_id = 1024
        self.pad_token_id = 1025
        self.rope_theta = 10000.0
        self.rope_scaling = None
        self.delay_pattern = [0, 8, 9, 10, 11, 12, 13, 14, 15]

class DiaConfigReduced:
    def __init__(self):
        self.model_type = "dia"
        self.is_encoder_decoder = True
        self.encoder_config = EncoderConfigReduced()
        self.decoder_config = DecoderConfigReduced()
        self.src_vocab_size = self.encoder_config.vocab_size
        self.tgt_vocab_size = self.decoder_config.vocab_size
        self.initializer_range = 0.02
        self.norm_eps = 1e-5
        self.torch_dtype = "float32"
        self.bos_token_id = 1026
        self.eos_token_id = 1024
        self.pad_token_id = 1025
        self.rope_theta = 10000.0
        self.rope_scaling = None
        self.delay_pattern = [0, 8, 9, 10, 11, 12, 13, 14, 15]

# --- Audio Utilities ---

def build_delay_indices(B: int, T: int, C: int, delay_pattern: tp.List[int]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32)
    t_idx_BxT = torch.broadcast_to(torch.arange(T, dtype=torch.int32)[None, :], [B, T])
    t_idx_BxTx1 = t_idx_BxT[..., None]
    t_idx_BxTxC = t_idx_BxTx1 - delay_arr.view(1, 1, C)
    b_idx_BxTxC = torch.broadcast_to(torch.arange(B, dtype=torch.int32).view(B, 1, 1), [B, T, C])
    c_idx_BxTxC = torch.broadcast_to(torch.arange(C, dtype=torch.int32).view(1, 1, C), [B, T, C])
    t_clamped_BxTxC = torch.clamp(t_idx_BxTxC, 0, T - 1)
    indices_BTCx3 = torch.stack([
        b_idx_BxTxC.reshape(-1),
        t_clamped_BxTxC.reshape(-1),
        c_idx_BxTxC.reshape(-1),
    ], dim=1).long()
    return t_idx_BxTxC, indices_BTCx3

def apply_audio_delay(audio_BxTxC: torch.Tensor, pad_value: int, bos_value: int, precomp: tp.Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    device = audio_BxTxC.device
    t_idx_BxTxC, indices_BTCx3 = precomp
    t_idx_BxTxC = t_idx_BxTxC.to(device)
    indices_BTCx3 = indices_BTCx3.to(device)
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.shape)
    mask_bos = t_idx_BxTxC < 0
    mask_pad = t_idx_BxTxC >= audio_BxTxC.shape[1]
    bos_tensor = torch.tensor(bos_value, dtype=audio_BxTxC.dtype, device=device)
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)
    result_BxTxC = torch.where(mask_bos, bos_tensor, torch.where(mask_pad, pad_tensor, gathered_BxTxC))
    return result_BxTxC

# --- Core Layers ---

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class MlpBlock(nn.Module):
    def __init__(self, embed_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, embed_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation in PyTorch."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [batch_size, num_heads, seq_len, head_dim]
        # position_ids: [batch_size, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Force float32 for better precision
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    # q, k: [batch_size, num_heads, seq_len, head_dim]
    # cos, sin: [batch_size, seq_len, head_dim]
    cos = cos.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, config: EncoderConfig | DecoderConfig, is_cross_attention=False):
        super().__init__()
        self.config = config
        self.is_cross_attention = is_cross_attention
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if is_cross_attention:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(config.cross_hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.cross_hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, self.max_position_embeddings)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: tp.Optional[torch.Tensor] = None,
        past_key_value: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: tp.Optional[torch.Tensor] = None,
        position_ids: tp.Optional[torch.LongTensor] = None,
        is_causal: bool = False,
    ) -> tp.Tuple[torch.Tensor, tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.is_cross_attention:
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(encoder_hidden_states).view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(encoder_hidden_states).view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        else:
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Only apply rotary embeddings for non-cross-attention
        if not self.is_cross_attention:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states)

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

class EncoderLayer(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MlpBlock(self.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tp.Tuple[torch.Tensor, tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value

class DecoderLayer(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MlpBlock(self.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.norm_eps)
        self.cross_attention = Attention(config, is_cross_attention=True)
        self.pre_cross_attention_layernorm = RMSNorm(self.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_attention_past_key_value: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tp.Tuple[torch.Tensor, tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]], tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            is_causal=True,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_cross_attention_layernorm(hidden_states)
        hidden_states, cross_attention_present_key_value = self.cross_attention(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            past_key_value=cross_attention_past_key_value,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value, cross_attention_present_key_value

# --- Main Models ---

class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.padding_idx = 0
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, input_ids, attention_mask, position_ids, past_key_values=None):
        hidden_states = self.embed_tokens(input_ids)
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        next_kv_cache = []
        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[idx],
            )
            hidden_states = layer_outputs[0]
            next_kv_cache.append(layer_outputs[1])
        hidden_states = self.norm(hidden_states)
        return hidden_states, next_kv_cache

class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.padding_idx = 1025
        self.vocab_size = config.vocab_size
        self.num_channels = config.num_channels
        self.embed_tokens = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            for _ in range(self.num_channels)
        ])
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size * self.num_channels, bias=False)

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        attention_mask,
        encoder_attention_mask,
        position_ids,
        past_key_values=None,
        cross_attention_past_key_values=None,
    ):
        hidden_states = None
        for i in range(self.num_channels):
            channel_embed = self.embed_tokens[i](input_ids[..., i])
            if hidden_states is None:
                hidden_states = channel_embed
            else:
                hidden_states += channel_embed

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        if cross_attention_past_key_values is None:
            cross_attention_past_key_values = [None] * len(self.layers)

        next_kv_cache = []
        next_cross_kv_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[idx],
                cross_attention_past_key_value=cross_attention_past_key_values[idx],
            )
            hidden_states = layer_outputs[0]
            next_kv_cache.append(layer_outputs[1])
            next_cross_kv_cache.append(layer_outputs[2])

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        logits = logits.view(*logits.shape[:-1], self.num_channels, self.vocab_size)
        return logits, next_kv_cache, next_cross_kv_cache

class DiaModel(nn.Module):
    def __init__(self, config: DiaConfig):
        super().__init__()
        self.encoder = Encoder(config.encoder_config)
        self.decoder = Decoder(config.decoder_config)

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        encoder_outputs=None,
        past_key_values=None,
        cross_attention_past_key_values=None,
    ):
        if encoder_outputs is None:
            encoder_position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=encoder_position_ids,
            )
        encoder_hidden_states = encoder_outputs[0]

        decoder_position_ids = torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device).unsqueeze(0).expand(decoder_input_ids.shape[0], -1)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            position_ids=decoder_position_ids,
            past_key_values=past_key_values,
            cross_attention_past_key_values=cross_attention_past_key_values,
        )
        return decoder_outputs[0], encoder_outputs, decoder_outputs[1], decoder_outputs[2]

if __name__ == '__main__':
    import gc
    # This is a simple example of how to use the model.
    # It is not a complete implementation of the inference logic.
    from torchpp.debug.flamegraph import profile_model_flamegraph   
    from torchpp.debug.benchmark import benchmark_model
    from torchpp.debug.model_info import count_params
    device = torch.device("cuda")
    config = DiaConfig()
    model = DiaModel(config).to(device) 
    # compiled_model = torch.compile(model , mode="max-autotune-no-cudagraphs")
    model.eval()
    # compiled_model.eval()

    # from torch.fx import symbolic_trace
    # graph = symbolic_trace(model)
    # print(graph.graph)

    # count_params(model)

    

    # # Example inputs
    text_tokens = torch.randint(0, config.encoder_config.vocab_size, (1, 100)).to(device)
    audio_tokens = torch.randint(0, config.decoder_config.vocab_size, (1, 50, config.decoder_config.num_channels)).to(device)
    
    # # Forward pass
    # with torch.no_grad():
    #     logits, _, _, _ = model(
    #         input_ids=text_tokens,
    #         attention_mask=torch.ones_like(text_tokens),
    #         decoder_input_ids=audio_tokens,
    #         decoder_attention_mask=torch.ones_like(audio_tokens[..., 0]),
    #     )

    # print("Logits shape:", logits.shape)
    # print("Success! Model runs without errors.")
    # benchmark_model(model, (text_tokens , torch.ones_like(text_tokens) , audio_tokens , torch.ones_like(audio_tokens[..., 0])) , steps=10)
    # benchmark_model(compiled_model, (text_tokens , torch.ones_like(text_tokens) , audio_tokens , torch.ones_like(audio_tokens[..., 0])) , steps=100)
    # profile_model_flamegraph(model , (text_tokens, torch.ones_like(text_tokens), audio_tokens, torch.ones_like(audio_tokens[..., 0])), out_dir="prof_out", warmup=2, steps=5)
    # Expected output: Logits shape: torch.Size([1, 50, 9, 1028])
    def benchmark():
        for i in range(1,10 ):
            text_len = i * 20
            audio_len = i * 10
            model = DiaModel(config).to(device)
            model.eval()
            text_tokens = torch.randint(0, config.encoder_config.vocab_size, (1, text_len)).to(device)
            audio_tokens = torch.randint(0, config.decoder_config.vocab_size, (1, audio_len, config.decoder_config.num_channels)).to(device) 
            benchmark_model(model, (text_tokens , torch.ones_like(text_tokens) , audio_tokens , torch.ones_like(audio_tokens[..., 0])) , steps=10) 
            del model
            del text_tokens
            del audio_tokens
            gc.collect()
            torch.cuda.empty_cache()


    benchmark()