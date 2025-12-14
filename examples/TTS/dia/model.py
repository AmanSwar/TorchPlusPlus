import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
from dataclasses import dataclass


from torchpp.dlops.normalization import RmsNormFused
from torchpp.dlops.linear import LinearSILU
from torchpp.attention.gqa import GroupedQueryAttention
from torchpp.attention.ca import CrossAttention
from torchpp.dlops.rope import compute_rope_params


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


class MlpBlock(nn.Module):
    def __init__(self, embed_dim : int , interm_dim : int , dtype=torch.float16):
        
        super().__init__()
        # self.fused_gate_proj = LinearSILU(in_features=embed_dim , out_features=interm_dim)
        self.gate_proj = nn.Linear(embed_dim , interm_dim , bias=False , dtype=dtype)
        self.activation = nn.SiLU()
        self.up_proj = nn.Linear(embed_dim , interm_dim , bias=False , dtype=dtype)
        self.down_proj = nn.Linear(interm_dim , embed_dim , bias=False , dtype=dtype)

    def forward(self , x : torch.Tensor):
        gate_out = self.activation(self.gate_proj(x))
        return self.down_proj(gate_out * self.up_proj(x))


class EncoderLayer(nn.Module):
    def __init__(self, config: EncoderConfig , dtype=torch.float16):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GroupedQueryAttention(
            d_in=config.hidden_size,
            num_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            qk_norm=False,
            dtype=dtype,
            # DEBUG=True
        )
        self.mlp = MlpBlock(self.hidden_size, config.intermediate_size , dtype=dtype)
        self.input_layernorm = RmsNormFused(self.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = RmsNormFused(self.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos , sin , kv_cache : tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(hidden_states , cos , sin , kv_cache)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        # print(f"Encoder Layer output shape: {hidden_states.shape}")
        return hidden_states 
    


class DecoderLayer(nn.Module):
    def __init__(self, config: DecoderConfig , dtype=torch.float16):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = config.hidden_size
        self.self_attn = GroupedQueryAttention(
            d_in=config.hidden_size,
            num_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            qk_norm=False,
            dtype=dtype,
            # DEBUG=True
        )
        self.mlp = MlpBlock(self.hidden_size, config.intermediate_size , dtype=dtype)

        self.input_layernorm = RmsNormFused(self.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = RmsNormFused(self.hidden_size, eps=config.norm_eps)
        self.cross_attention = CrossAttention(
            embed_dim=config.hidden_size,
            cross_dim=config.cross_hidden_size,
            n_heads=config.cross_num_attention_heads,
            qknorm=False,
            dtype=dtype
        )
        self.pre_cross_attention_layernorm = RmsNormFused(self.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cos : torch.Tensor,
        sin : torch.Tensor,
        kv_cache: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_kv_cache: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None,

    ) -> torch.Tensor:
        if(hidden_states.dtype != self.dtype):
            hidden_states = hidden_states.to(self.dtype)
        if(encoder_hidden_states.dtype != self.dtype):
            encoder_hidden_states = encoder_hidden_states.to(self.dtype)

        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cos,
            sin,
            kv_cache,

        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_cross_attention_layernorm(hidden_states)
        
        hidden_states = self.cross_attention(
            x=hidden_states,
            y=encoder_hidden_states,
            kv_cache=cross_kv_cache
        )


        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        # print(f"Decoder Layer output shape: {hidden_states.shape}")
        return hidden_states
    

class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig , dtype=torch.float16):
        super().__init__()
        self.config = config
        self.padding_idx = 0
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx , dtype=dtype)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RmsNormFused(config.hidden_size, eps=config.norm_eps)
        
        self.cos , self.sin = compute_rope_params(
            head_dim=config.head_dim,
            theta_base=1e6,
            context_length=config.max_position_embeddings,
            device="cuda",
            dtype=dtype,
        )

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)

        N = input_ids.shape[-1]
        self.cos, self.sin = self.cos[:N], self.sin[:N]

        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                self.cos,
                self.sin,
            )
            hidden_states = layer_outputs
        hidden_states = self.norm(hidden_states)
        return hidden_states
    

class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig , dtype=torch.float16):
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
        self.norm = RmsNormFused(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size * self.num_channels, bias=False , dtype=dtype)
        self.cos , self.sin = compute_rope_params(
            head_dim=config.head_dim,
            theta_base=1e6,
            context_length=config.max_position_embeddings,
            device="cuda",
            dtype=dtype,
        )
    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        
    ):
        # print(f"\nInside Decoder forward. input_ids shape: {input_ids.shape}")
        hidden_states : torch.Tensor = None
        
        for i in range(self.num_channels):
            channel_embed = self.embed_tokens[i](input_ids[..., i])
            if hidden_states is None:
                hidden_states = channel_embed
            else:
                hidden_states += channel_embed

        # if(hidden_states.dim() != 3):
        #     hidden_states = hidden_states.unsqueeze(0)

        N = input_ids.shape[-2]

        self.cos, self.sin = self.cos[:N], self.sin[:N]
       
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                encoder_hidden_states,
                self.cos,
                self.sin,
            )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        logits = logits.view(*logits.shape[:-1], self.num_channels, self.vocab_size)
        # print(f"\nDecoder output logits shape: {logits.shape}")
        return logits
    


class DiaModel(nn.Module):
    def __init__(self, config: DiaConfig , dtype=torch.float16):
        super().__init__()
        self.encoder = Encoder(config.encoder_config ,dtype=dtype)
        self.decoder = Decoder(config.decoder_config ,dtype=dtype)

    def forward(
        self,
        input_ids,
        decoder_input_ids,
        encoder_outputs=None,
    ):
        if encoder_outputs is None:
            encoder_position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
            encoder_outputs = self.encoder(
                input_ids=input_ids,
            )
        encoder_hidden_states = encoder_outputs[0]

        decoder_position_ids = torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device).unsqueeze(0).expand(decoder_input_ids.shape[0], -1)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        # print(f"\nFinal DiaModel output (decoder_outputs) logits shape: {decoder_outputs.shape}")
        return decoder_outputs, encoder_outputs, None , None
    
if __name__ == "__main__":
    
    from torchpp.debug.benchmark import benchmark_model
    from torchpp.debug.model_info import count_params

    device = torch.device("cuda")

    config = DiaConfig()
    # model = DiaModel(config).to(device)
    # model.eval()

    # count_params(model)

    def benchmark():
        for i in range(1,10 ):
            text_len = i * 20
            audio_len = i * 10
            model = DiaModel(config).to(device)
            model.eval()
            text_tokens = torch.randint(0, config.encoder_config.vocab_size, (1, text_len)).to(device)
            audio_tokens = torch.randint(0, config.decoder_config.vocab_size, (1, audio_len, config.decoder_config.num_channels)).to(device)
            benchmark_model(model, (text_tokens , audio_tokens) , steps=10)
    # # Example inputs
    # text_tokens = torch.randint(0, config.encoder_config.vocab_size, (1, 100)).to(device)
    # audio_tokens = torch.randint(0, config.decoder_config.vocab_size, (1, 50, config.decoder_config.num_channels)).to(device)
    
    # # Forward pass
    # with torch.no_grad():
    #     logits, _, _, _ = model(
    #         input_ids=text_tokens,
    #         decoder_input_ids=audio_tokens,
    #     )

    # print("Logits shape:", logits.shape)
    # print("Success! Model runs without errors.")
    
    # benchmark_model(model, (text_tokens , audio_tokens) , steps=10)
    benchmark()
