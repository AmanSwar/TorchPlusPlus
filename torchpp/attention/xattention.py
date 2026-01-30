"""

Main aim is to make a generalized attention backend which can be extended to any 
attention type

Attention mainly is 

sigma(QK/root(dk)).V

The Attention class will perform exactly that

"""



import torch
import torch.nn as nn

from einops import rearrange , pack, unpack

from typing import Literal

from torchpp.types import Int , Bool , Tensor
from torchpp.dlops.linear import LinearNBFp16 , LinearNBBf16
from torchpp.utils import default





# to do : make fused kernels for qkv projection for fp16 and bf16\
# fused kernels -> qkv projection + qk normalization
class QKV(nn.Module):

  def __init__(
      self,
      in_dimension :  int,   # int not Int (cannot be none)
      head_dim : Int = None,
      num_heads : Int = None,
      q_head_dim : Int = None,
      k_head_dim : Int = None,
      v_head_dim : Int = None,
      num_q_heads : Int = None,
      num_k_heads : Int = None,
      num_v_heads : Int = None,
      qk_normalize : Literal["layernorm" , "rmsnorm" , None]  = None,
      dtype : torch.dtype = torch.float16

  ):
    super().__init__()

    assert num_heads != None or num_q_heads != None, \
      "Specify number of heads"
    
    assert not (head_dim != None and (q_head_dim != None or k_head_dim != None or v_head_dim != None)) , \
      "Either provide head_dim or individual head dims"

    self.in_dim = in_dimension
    
    self.q_head_dim = default(q_head_dim , head_dim)
    self.k_head_dim = default(k_head_dim , head_dim)
    self.v_head_dim = default(v_head_dim , head_dim)

    self.num_q_heads = default(num_q_heads , num_heads)
    self.num_k_heads = default(num_k_heads , num_heads)
    self.num_v_heads = default(num_v_heads , num_heads)

    self.qk_normalize = qk_normalize
    
    if dtype == torch.float16:
      self.Wq = LinearNBFp16(
        in_features= self.in_dim,
        out_features = self.num_heads * self.q_head_dim
      )

      self.Wk = LinearNBFp16(
        in_features= self.in_dim,
        out_features=self.num_heads * self.k_head_dim
      )  

      self.Wv = LinearNBFp16(
        in_features=self.in_dim,
        out_features=self.num_heads * self.v_head_dim
      )

    if dtype == torch.bfloat16: 
      self.Wq = LinearNBBf16(
        in_features= self.in_dim,
        out_features = self.num_heads * self.q_head_dim
      )

      self.Wk = LinearNBBf16(
        in_features= self.in_dim,
        out_features=self.num_heads * self.k_head_dim
      )  

      self.Wv = LinearNBBf16(
        in_features=self.in_dim,
        out_features=self.num_heads * self.v_head_dim
      )

    if(self.qk_normalize):
      if(self.qk_normalize == "layernorm"):
        self.q_norm = nn.LayerNorm(self.q_head_dim , eps=1e-6 ,dtype=dtype)
        self.k_norm = nn.LayerNorm(self.k_head_dim , eps=1e-6 ,dtype=dtype)
      
      if(self.qk_normalize == "rmsnorm"):
        self.q_norm = nn.RMSNorm(self.q_head_dim, eps=1e-6 ,dtype=dtype)
        self.k_norm = nn.RMSNorm(self.k_head_dim, eps=1e-6 ,dtype=dtype)
    
  def forward(
      self,
      query_input : torch.Tensor,
      key_input : Tensor = None,
      value_input : Tensor = None
  ):

    key_input = default(key_input , query_input)
    value_input = default(value_input , query_input)

    Q = self.Wq(query_input)
    K = self.Wk(key_input)
    V = self.Wv(value_input)


    Q = rearrange(Q , 'b n (h d) -> b h n d' , h = self.num_q_heads)
    K = rearrange(K , 'b n (h d) -> b h n d' , h = self.num_k_heads)
    V = rearrange(V , 'b n (h d) -> b h n d' , h = self.num_v_heads)

    if(self.qk_normalize):
      Q = self.q_norm(Q)
      K = self.k_norm(K)
    
    return Q , K , V


# class XAttention(nn.Module):

#   def __init__(
#       self,
#       in_dimension : int,    # int not Int (cannot be none)
#       head_dim : Int = None,
#       num_heads : Int = None,
#       num_q_heads : Int = None,
#       num_k_heads : Int = None,
#       num_v_heads : Int = None,
#       q_head_dim : Int = None,
#       k_head_dim : Int = None,
#       v_head_dim : Int = None,
#       qk_normalize : Bool = None,
#       dtype : torch.dtype = torch.float16

#   ):
    
#     super().__init__()

#     assert num_heads != None or num_q_heads != None, \
#       "Specify number of heads"
    
#     if(num_q_heads):
#       assert num_q_heads % num_k_heads == 0 & num_q_heads % num_v_heads == 0, \
#         "number of Q heads must be divisible by K and V heads"
      
#     assert dtype == torch.float16 or dtype == torch.bfloat16 , \
#       "For using flashattention as backend , must use fp16 or bf16"
    
#     self.in_dim = in_dimension
#     self.out_dim = out_dimension

#     self.q_dim = self.out_dim // default(num_heads , num_q_heads)
#     self.k_dim = self.out_dim // default(num_heads , num_k_heads)
#     self.v_dim = self.out_dim // default(num_heads , num_v_heads)

#     self.qkv_projection = QKV(
#       in_dimension = self.in_dim,
#       out_dimension = self.out_dim,
#       num_heads = default(num_heads , num_q_heads),
#       q_head_dim = self.q_dim,
#       k_head_dim = self.k_dim,
#       v_head_dim = self.v_dim,
#       dtype = dtype
#     )


#     self.qk_normalize = qk_normalize

#     if self.qk_normalize:
#       self.q_norm = nn.RMSNorm(self.out_dim // default(num_heads , num_q_heads), eps=1e-6 ,dtype=dtype)
#       self.k_norm = nn.RMSNorm(self.out_dim // default(num_heads , num_k_heads), eps=1e-6 ,dtype=dtype)
    
#   def forward(
#       self,
#       query_input : Tensor,
#       key_input : Tensor = None,
#       value_input : Tensor = None
#   ):
    
    
#     key_input = default(key_input , query_input)
#     value_input = default(value_input , query_input) 

#     Q , K , V = self.qkv_projection(
#       query_input = query_input,
#       key_input = key_input,
#       value_input = value_input
#     )

    

   
    

    