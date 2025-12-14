import torch
import torch.nn as nn

from typing import Optional 
from torchpp.llmops.kvcache import KVCache

from flash_attn import flash_attn_func


class CrossAttention(nn.Module):

  def __init__(
      self, 
      embed_dim : int,
      cross_dim : int,
      n_heads : int,
      qknorm : bool = True,
      dtype : torch.dtype = torch.float16,
  ):
    #all consts   
    super().__init__() 
    self.embed_dim = embed_dim
    self.cross_dim = cross_dim
    self.n_heads = n_heads
    
    assert embed_dim % n_heads == 0, "embed dimension should be divisible by number of heads"

    self.head_dim = int(embed_dim / n_heads)
    
    self.qknorm = qknorm
    self.dtype = dtype


    self.Wq = nn.Linear(
      self.embed_dim,
      self.embed_dim,
      bias=False,
      dtype=self.dtype
    )

    self.Wk = nn.Linear(
      self.cross_dim,
      self.embed_dim,
      bias=False,
      dtype=self.dtype
    )

    self.Wv = nn.Linear(
      self.cross_dim,
      self.embed_dim,
      bias=False,
      dtype=self.dtype
    )

    self.out_projection = nn.Linear(self.embed_dim , self.embed_dim , dtype=self.dtype)

    
    
  def forward(
      self ,
      x : torch.Tensor , 
      y : torch.Tensor,
      kv_cache : Optional[KVCache] = None,
  ) -> torch.Tensor:

    batch_size , seq_line , _ = x.shape

    Q : torch.Tensor = self.Wq(x)
    K : torch.Tensor = self.Wk(y)
    V : torch.Tensor = self.Wv(y)


    Q = Q.view(batch_size , seq_line , self.n_heads , self.head_dim)
    K = K.view(batch_size , seq_line , self.n_heads , self.head_dim)
    V = V.view(batch_size , seq_line , self.n_heads , self.head_dim)

    #update kv cache
    if kv_cache is not None:
      K , V = kv_cache.update(K , V)  

    attention_out : torch.Tensor | None = flash_attn_func(Q , K , V)

    attention_out = attention_out.reshape(batch_size , seq_line , self.embed_dim)

    return self.out_projection(attention_out)
