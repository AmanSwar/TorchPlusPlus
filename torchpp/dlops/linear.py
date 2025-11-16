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
  
#inspired from jax
class Dense(nn.Module):
  
  def __init__(
      self,
      in_featrues : tuple[int , ...],
      out_features : tuple[int , ...],
      axis : tuple[int , ...] = (-1,),#default to last dim (like nn.Linear)
      weight_dtype : torch.dtype = torch.float16,
      device : torch.device = torch.device("cuda")
  ):
    """
    dense layer for doing matmul in arbitrary axes

    Args:
        in_featrues : tuple[int , ...] = input feature dimensions
        out_features : tuple[int , ...] = output feature dimensions
        axis : tuple[int , ...] = axes to perform the dense operation on
        weight_dtype : torch.dtype = dtype for weight parameter
        device : torch.device = device for weight parameter
    """
    super().__init__()

    self.in_features = in_featrues
    self.out_features = out_features
    self.axis = axis
    self.kernel_shape = self.in_features + self.out_features

    self.weight = nn.Parameter(
      torch.empty(
        self.kernel_shape,
        dtype=weight_dtype,
        device=device
      )
    )

    nn.init.normal_(self.weight , std=0.02)

  def _normalize_axes(self , axes : tuple[int , ...] , ndim : int) -> tuple[int , ...]:
    return tuple(
      axis if axis >=0 else ndim + axis
      for axis in axes
    )
  

  def forward(self , x : torch.Tensor) -> torch.Tensor:
    norm_axis = self._normalize_axes(self.axis , x.ndim)
    kernel_contract_axes = tuple(range(len(norm_axis)))
    output = torch.tensordot(
      x.to(self.weight.dtype),
      self.weight,
      dims=(norm_axis , kernel_contract_axes)
    ).to(x.dtype)
    
    return output
   