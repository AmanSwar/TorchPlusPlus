import os
import json
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_model_flamegraph(
    model: torch.nn.Module,
    example_inputs,
    out_dir: str = "profiler_output",
    warmup: int = 3,
    steps: int = 10,
    profile_cuda: bool = True,
    record_shapes: bool = True,
    with_stack: bool = True,
    device: str | torch.device = None,
):
    """
    Profiles `model` run on example_inputs and writes:
      - out_dir/trace.json      (Chrome trace)
      - out_dir/flamegraph.html (interactive speedscope page that loads trace.json)

    Returns (html_path, trace_path).

    example_inputs: a single tensor, or a tuple/list of tensors (same call signature you'd pass to model).
    device: 'cuda' or 'cpu' or torch.device. If None, uses CUDA if available.
    """
    # Normalize device
    if device is None:
        device = torch.device("cuda" if (profile_cuda and torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(device)

    # Prepare output dir
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, f"{model.__class__.__name__} trace.json")
    html_path = os.path.join(out_dir, "flamegraph.html")

    # Put model on device
    model = model.to(device)
    model.eval()

    # Normalize example_inputs into tuple
    if isinstance(example_inputs, (list, tuple)):
        inputs = tuple(inp.to(device) for inp in example_inputs)
    else:
        # single tensor
        inputs = (example_inputs.to(device),)

    # Warmup runs (no profiler) to reduce first-run overhead noise
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)

    # Choose activities
    activities = [ProfilerActivity.CPU]
    if profile_cuda and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Run profiler
    with profile(
        activities=activities,
        record_shapes=record_shapes,
        with_stack=with_stack,
        profile_memory=False,
    ) as prof:
        for _ in range(steps):
            # small wrapper so you get a named region in the trace
            with record_function("model_infer"):
                _ = model(*inputs)
                # If you want backward profiling, call backward here (and zero grads etc).
                # e.g. loss.backward() -- not included by default.

    # Export to Chrome trace format
    prof.export_chrome_trace(trace_path)

    
    return trace_path
