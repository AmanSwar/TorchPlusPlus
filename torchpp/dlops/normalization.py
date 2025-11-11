import torch
import torch.nn as nn

import layernormFused
import rmsnormFused

class LayerNormFused(nn.Module):

  def __init__(
      self,
      normalize_dim_shape : int,
      eps : float = 1e-6,
      dtype : torch.dtype = torch.float16,
      device : torch.device = torch.device("cuda") #compulsory
  ):
    
    super().__init__()

    self.shape = normalize_dim_shape
    self.eps = eps

    self.weight = nn.Parameter(
      torch.ones(self.shape , dtype=dtype , device=device)
    )

  def forward(self , x : torch.Tensor):
    return layernormFused.applyLayernorm(x , self.weight , self.eps)


class RmsNormFused(nn.Module):

  def __init__(
      self,
      normalize_dim_shape : int,
      eps : float = 1e-6,
      dtype : torch.dtype = torch.float16,
      device : torch.device = torch.device("cuda")
  ):
    
    super().__init__()

    self.shape = normalize_dim_shape
    self.eps = eps

    self.weight = nn.Parameter(
      torch.ones(self.shape , dtype=dtype , device=device)
    )

  def forward(self,  x : torch.Tensor):
    return rmsnormFused.rmsnormFused(x , self.weight, self.eps)

