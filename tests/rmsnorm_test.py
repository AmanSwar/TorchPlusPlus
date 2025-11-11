import torch
import torch.nn as nn


from torchpp.dlops.normalization import RmsNormFused


if __name__ == "__main__":
    print("Testing LayerNorm CUDA implementation...")
    
    batch_size = 32
    seq_len = 128
    hidden_dim = 512
    
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
    
    custom_ln = RmsNormFused(hidden_dim, eps=1e-6).cuda()
    
    torch_ln = nn.RMSNorm(hidden_dim, eps=1e-6).cuda().half()
    torch_ln.weight.data = custom_ln.weight.data.clone()
    
    with torch.no_grad():
        custom_output = custom_ln(x)
        torch_output = torch_ln(x)
    
    max_diff = (custom_output - torch_output).abs().max().item()
    mean_diff = (custom_output - torch_output).abs().mean().item()
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    print(f"Output shape: {custom_output.shape}")
    
    if max_diff < 1e-2:  # Tolerance for fp16
        print("Test passed!")
    else:
        print("Test failed - large difference detected")