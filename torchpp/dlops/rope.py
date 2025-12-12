import torch

import ropeApplyFunction
import qkropeApplyFunction

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


def qkrope_apply(
  q : torch.Tensor,
  k : torch.Tensor,
  cos : torch.Tensor,
  sin : torch.Tensor
):
  """
  applies RoPE to Q and K
  Args:
      q (torch.Tensor) : query tensor
      k (torch.Tensor) : key tensor
      cos (torch.Tensor): cos cache
      sin (torch.Tensor): sin cache

  Returns:
      Tuple[torch.Tensor, torch.Tensor]
  """
  return qkropeApplyFunction.qkrope_apply_cuda(q , k , cos , sin)