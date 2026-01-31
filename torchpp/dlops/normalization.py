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




# ========================== Unfused Ops ========================== #


class LayerNorm4D(nn.Module):

  def __init__(
      self,
      num_features : int,
  ):
    
    super().__init__()

    self.gamma = nn.Parameter(torch.ones(1 , 1 , 1 , num_features))
    self.beta = nn.Parameter(torch.zeros(1 , 1 , 1 , num_features))

  def forward(self , x : torch.Tensor):

    mean = x.mean(dim=[2,3] , keepdim=True)
    var = x.var(dim=[2,3] , keepdim=True , unbiased=False)

    x_norm = (x - mean) / torch.sqrt(var + 1e-6)

    return x_norm * self.gamma + self.beta
  

