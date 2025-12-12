import ropeApplyFunction

import torch

from torchpp.dlops.rope import qkrope_apply

def rope_apply(
  x : torch.Tensor,
  cos : torch.Tensor,
  sin : torch.Tensor
):
  return ropeApplyFunction.rope_apply_cuda(x , cos , sin)


def compute_rope_params(
    head_dim, theta_base, context_length, device="cuda", dtype=torch.float16
):
    assert head_dim % 2 == 0, "head dim must be divisible by 2"
    ar = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inv_freq = 1.0 / (theta_base ** (ar / head_dim))
    pos = torch.arange(context_length, device=device, dtype=dtype)
    angles = pos[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)  # [context_length, head_dim]
    cos = torch.cos(angles).to(torch.float16).contiguous()
    sin = torch.sin(angles).to(torch.float16).contiguous()
    return cos, sin


bs = 1
n_heads = 16
seq_len = 256
head_dim = 128
device = torch.device("cuda")

cos , sin = compute_rope_params(head_dim , 1e6 , seq_len)
x = torch.rand(bs , n_heads , seq_len , head_dim , device=device , dtype=torch.float16) # query tensor
y = torch.rand(bs , n_heads , seq_len , head_dim , device=device , dtype=torch.float16) # key tensor 
# out = rope_apply(x , cos , sin)
out = qkrope_apply(x , y , cos , sin)

print(out)

