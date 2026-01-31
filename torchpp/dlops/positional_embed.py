import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):

  def __init__(
      self,
      seq_len : int,
      embed_dim : int,
      dropout : float = 0.1,
  ):
    
    super().__init__()

    self.embed_dim = embed_dim
    self.seq_len = seq_len

    self.position_embeddings = nn.Parameter(torch.zeros(seq_len , seq_len))
    self.layer_norm = nn.LayerNorm(embed_dim + seq_len)
    self.dropout = nn.Dropout(dropout)


    nn.init.trunc_normal_(self.position_embeddings , std=0.02)


  def forward(self , x , train=True):

    batch_size , seq_len , _ = x.shape

    pos_embed = self.position_embeddings[:seq_len , :seq_len].unsqueeze(0)
    pos_embed = pos_embed.expand(batch_size , -1 , -1)

    output = torch.cat([x , pos_embed] , dim=-1)
    output = self.layer_norm(output)
    output = self.dropout(output)
    
    return output