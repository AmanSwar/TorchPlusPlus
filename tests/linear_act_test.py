import linearActvationFp16

import torch

batch_size = 32
input_dim = 512
output_dim = 256

input = torch.randn(batch_size, input_dim, dtype=torch.float16, device='cuda')
weight = torch.randn(output_dim, input_dim, dtype=torch.float16, device='cuda')

output = linearActvationFp16.LinearSilu(input, weight)


print(f"Input shape: {input.shape}")
print(f"Weight shape: {weight.shape}")
print(f"Output shape: {output.shape}")