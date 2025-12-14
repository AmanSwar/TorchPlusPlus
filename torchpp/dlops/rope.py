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

def compute_rope_params(
    head_dim, theta_base, context_length, device="cuda", dtype=torch.float32
):
    assert head_dim % 2 == 0, "head dim must be divisible by 2"
    ar = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inv_freq = 1.0 / (theta_base ** (ar / head_dim))
    pos = torch.arange(context_length, device=device, dtype=dtype)
    angles = pos[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles).to(torch.float16).contiguous()
    sin = torch.sin(angles).to(torch.float16).contiguous()
    return cos, sin

