import torch
import torch.nn as nn


import linearActvationFp16


class LinearGELU(nn.Module):

  def __init__(self , in_features , out_features):
    """
    Linear layer with fused GELU activation

    Args:
        in_features : int = input dim
        out_features : int = output dim
    """
    super().__init__()

    self.weight = nn.Parameter(
      torch.randn(out_features , in_features , dtype=torch.float16)
    )

  def forward(self , x):
    return linearActvationFp16.LinearGelu(x , self.weight)
  

class LinearSILU(nn.Module):

  def __init__(self , in_features , out_features):
    """
    Linear layer with fused SILU activation

    Args:
        in_features : int = input dim
        out_features : int = output dim
    """
    super().__init__()

    self.weight = nn.Parameter(
      torch.randn(out_features , in_features , dtype=torch.float16)
    )

  def forward(self , x):
    return linearActvationFp16.LinearSilu(x , self.weight)
  
  
   