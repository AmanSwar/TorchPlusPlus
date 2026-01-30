"""
Spatio-Temporal Transformer implementation

input -> images or videos
image -> (b , c , h , w)
videos -> (b , c , t , h , w) | (b , t , h , w ,c)

Spatial 


"""


import torch
import torch.nn as nn

from einops import rearrange , repeat , pack , unpack

from flash_attn import flash_attn_func

from torchpp.dlops.linear import LinearNBFp16 , LinearNBBf16
from torchpp.attention import QKV
from torchpp.utils import exists

class SpatialTransformer(nn.Module):
  """
  Spatial Tranformer keeps attention amoung the pixed for images and videos
  

  Args:
      nn (_type_): _description_
  """
  
  def __init__(
      self,
      in_dimension : int,
      head_dimension : int,
      num_q_heads : int,
      num_k_heads : int,
      num_v_heads : int,
      qk_norm : bool,
      transpose : bool = False,
      dtype : torch.dtype = torch.float16
  
  ):
    super().__init__()

    assert num_q_heads % num_k_heads == 0 & num_q_heads % num_v_heads == 0, \
      "Number of Q heads must be divisble by K and V heads" 
    
    assert dtype == torch.float16 or dtype == torch.bfloat16 , \
    "Dtype must be float16 or bfloat16"
    
    self.in_dim = in_dimension
    self.head_dim = head_dimension
    self.num_q_heads = num_q_heads
    self.num_k_heads = num_k_heads
    self.num_v_heads = num_v_heads

    self.out_dimension_q = self.head_dim * self.num_q_heads
    self.out_dimension_k = self.head_dim * self.num_k_heads
    self.out_dimension_v = self.head_dim * self.num_v_heads
    

    if dtype == torch.float16:
      self.Wq = LinearNBFp16(
        in_features= self.in_dim,
        out_features = self.out_dimension_q
      )

      self.Wk = LinearNBFp16(
        in_features= self.in_dim,
        out_features=self.out_dimension_k
      )  

      self.Wv = LinearNBBf16(
        in_features=self.in_dim,
        out_features=self.out_dimension_v
      )

      self.Wo = LinearNBFp16(
        in_features= self.out_dimension_q,
        out_features= self.in_dim
      )

    if dtype == torch.bfloat16: 
      self.Wq = LinearNBBf16(
        in_features= self.in_dim,
        out_features = self.out_dimension_q
      )

      self.Wk = LinearNBBf16(
        in_features= self.in_dim,
        out_features=self.out_dimension_k
      )  

      self.Wv = LinearNBBf16(
        in_features=self.in_dim,
        out_features=self.out_dimension_v 
      )
      self.Wo = LinearNBBf16(
        in_features= self.out_dimension_q,
        out_features= self.in_dim
      )

    self.qkv_projection = QKV(
      in_dimension= self.in_dim,
      num_q_heads = self.num_q_heads,
      num_k_heads = self.num_k_heads,
      num_v_heads = self.num_v_heads,
      head_dim = self.head_dim,
      qk_normalize = qk_norm,
      dtype = dtype
    )
    self.transpose = transpose
  def forward(
      self,
      video_tensor : torch.Tensor, 
      cond : torch.Tensor | None = None,
  ):
    #currently not supporting kv cache
    # fixed cos and sin value (not on-fly)

    pattern = 'b c ... h w' if self.transpose else 'b ... h w c'
    inp = rearrange(video_tensor , f'{pattern} -> b ... h w c')

    b , *t , h , w , c = video_tensor.shape

    inp, t_ps = pack([inp], '* h w c')        
    inp, s_ps = pack([inp], 'b * c')

    cond = repeat(cond, 'b hw c -> (b t) hw c', t=t if exists(t) else 1) if exists(cond) else None 
    
    Q , K , V = self.qkv_projection(
      query_input = inp,
      key_input = cond if exists(cond) else inp,
    )


    Q = Q.transpose(1 , 2)  # b h n d -> b n h d
    K = K.transpose(1 , 2)
    V = V.transpose(1 , 2)

    attn_output : torch.Tensor | None = flash_attn_func(Q , K , V , causal = True)
    attn_output = rearrange(attn_output , 'b n h d -> b n (h d)')

    return self.Wo(attn_output).contiguous().view(b , *t_ps , h , w , c) if not self.transpose else self.Wo(attn_output).contiguous().view(b , *t_ps , h , w , c).transpose(-1 , -3)

    
  

class TemporalTransformer(nn.Module):
  pass


class ST_Transformer(nn.Module):
  pass