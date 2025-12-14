import torch

import time

def benchmark_model(
  model : torch.nn.Module,
  example_inputs,
  warmup: int = 5,
  steps : int = 10,
  device : str | torch.device = torch.device("cuda")
):
  
  model = model.to(device)
  model.eval()


  if isinstance(example_inputs, (list, tuple)):
    inputs = tuple(inp.to(device) for inp in example_inputs)
  else:
    # single tensor
    inputs = (example_inputs.to(device),)

  # Warmup runs (no profiler) to reduce first-run overhead noise
  with torch.no_grad():
    for _ in range(warmup):
        _ = model(*inputs)


  start = time.monotonic()

  for _ in range(steps):
    out = model(*inputs)

  final_time = time.monotonic() - start


  print("\nFinish Benchmarking !")
  print("\nResults:\n")
  print(f"Input Size : {[inp.shape for inp in inputs]}")
  print(f"Time Taken : {(final_time / steps):.3f}")
  print(f"Throughput  : {(1 / (final_time / steps)):.3f} samples/sec")
  print("\n----------------\n")


