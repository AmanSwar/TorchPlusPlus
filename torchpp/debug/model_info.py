import torch

def count_params(
   model : torch.nn.Module,
   logging : bool = True
):
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  
  if logging:
   print(f"Total parameters: {(total/1000000):.3f}M")
   print(f"Trainable parameters: {(trainable/1000000):.3f}M")
   print("\n")

  return total, trainable


