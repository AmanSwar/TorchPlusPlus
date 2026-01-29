import torch
import torch.nn as nn

from flash_attn import flash_attn_func

from torchpp.dlops.rope import rope_apply


class MultiQueryAttention(nn.Module):

  def __init__(
      self,
      num_q_heads : int,
      embed_dim : int, 
      qk_norm : bool = True,
      dtype : torch.dtype =torch.float16
  ):
    
    super().__init__()

    self.embed_dim = embed_dim
    self.num_q_head = num_q_heads
    self.head_dim = int(self.embed_dim / self.num_q_head)


    self.Wq = nn.Linear(
      self.embed_dim,
      self.embed_dim,
      dtype=dtype,
      bias=False
    )

    self.Wk = nn.Linear(
      self.embed_dim,
      self.head_dim,
      dtype=dtype,
      bias=False
    )

    self.Wv = nn.Linear(
      self.embed_dim,
      self.head_dim,
      dtype=dtype,
      bias=False
    )


    self.out_projection = nn.Linear(
      self.embed_dim,
      self.embed_dim,
      dtype=dtype,
      bias=False
    )
  
    self.qk_norm = qk_norm


    self.q_norm = nn.RMSNorm(self.head_dim , eps=1e-6)
    self.k_norm = nn.RMSNorm(self.head_dim , eps=1e-6)


  def forward(self , x , cos , sin):

    batch_size , seq_len , _ = x.shape

    Q : torch.Tensor = self.Wq(x) # [bs , seq_len , embed_dim]
    K : torch.Tensor = self.Wk(x) # [bs , seq_len , head_dim]
    V : torch.Tensor = self.Wv(x) # [bs , seq_len , head_dim]

    Q = Q.view(batch_size , seq_len , self.num_q_head , self.head_dim).transpose(1,2)
    K = K.view(batch_size , seq_len , 1 , self.head_dim).transpose(1,2)
    V = V.view(batch_size , seq_len , 1 , self.head_dim).transpose(1,2)


    if self.qk_norm:

      Q = self.q_norm(Q)
      K = self.k_norm(K)

    Q = rope_apply(Q , cos , sin)
    K = rope_apply(Q , cos , sin)

    Q = Q.transpose(1,2)
    K = K.transpose(1,2)
    V = V.transpose(1,2)

    attention_out : torch.Tensor | None = flash_attn_func(Q , K , V , causal=True)
    # logits = Q @ K.transpose(-2,-1)
    # softmax_logits = nn.functional.softmax(logits / sqrt(self.head_dim))
    # attention = softmax_logits @ V
    # attention = attention.reshape(batch_size , seq_len , self.embed_dim)
    
    return self.out_projection(attention_out.reshape(batch_size , seq_len , self.embed_dim))
  