import torch

import ropeApplyFunction


def rope_apply(
  x : torch.Tensor,
  cos : torch.Tensor,
  sin : torch.Tensor
):
  """
  applies RoPE
  Args:
      x (torch.Tensor) : input tensor
      cos (torch.Tensor): cos cache
      sin (torch.Tensor): sin cache

  Returns:
      torch.Tensor
  """
  return ropeApplyFunction.rope_apply_cuda(x , cos , sin)