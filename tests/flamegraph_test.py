# small example
import torch
import torchvision.models as models
from torchpp.debug.flamegraph import profile_model_flamegraph
device = torch.device("cuda")
model = models.resnet18(pretrained=False).to(device)
# single example input batch: batch size 8, 3x224x224
example = torch.randn(8, 3, 224, 224)

trace = profile_model_flamegraph(model, example, out_dir="prof_out", warmup=2, steps=5 , device=device)
print("Or load trace.json at speedscope.app:", trace)
