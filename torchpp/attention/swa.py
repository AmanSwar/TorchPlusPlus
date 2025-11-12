import torch
import torch.nn as nn

from flash_attn import flash_attn_func

from torchpp.dlops.rope import rope_apply

class SlidingWindowAttention(nn.Module):

  def __init__(
      self, 
      window_size : int,
      embed_dim : int,
      n_heads : int,
      qknorm : bool = True,
      dtype : torch.dtype = torch.float16,
  ):
    #all consts  
    self.window_size = window_size  
    self.embed_dim = embed_dim
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
      self.embed_dim,
      self.embed_dim,
      bias=False,
      dtype=self.dtype
    )

    self.Wv = nn.Linear(
      self.embed_dim,
      self.embed_dim,
      bias=False,
      dtype=self.dtype
    )

    self.out_projection = nn.Linear(self.embed_dim , self.embed_dim , dtype=self.dtype)

    if self.qknorm:
      self.q_norm = nn.RMSNorm(self.embed_dim)
      self.k_norm = nn.RMSNorm(self.embed_dim)
    
    
  def forward(self , x : torch.Tensor , cos , sin):
    batch_size , seq_line , _ = x.shape

    Q : torch.Tensor = self.Wq(x)
    K : torch.Tensor = self.Wk(x)
    V : torch.Tensor = self.Wv(x)


    Q = Q.view(batch_size , seq_line , self.n_heads , self.head_dim).transpose(1,2)
    K = K.view(batch_size , seq_line , self.n_heads , self.head_dim).transpose(1,2)
    V = V.view(batch_size , seq_line , self.n_heads , self.head_dim).transpose(1,2)


    if self.qknorm:

      Q = self.q_norm(Q)
      K = self.k_norm(K)

    Q = rope_apply(Q , cos , sin)
    K = rope_apply(K , cos , sin)


    Q = Q.transpose(1,2)
    K = K.transpose(1,2)
    V = V.transpose(1,2)

    attention_out : torch.Tensor | None = flash_attn_func(
      Q , K , V,
      causal=True,
      window_size=(self.window_size - 1 , 0)
    )

    attention_out = attention_out.reshape(batch_size , seq_line , self.embed_dim)

    return self.out_projection(attention_out)
