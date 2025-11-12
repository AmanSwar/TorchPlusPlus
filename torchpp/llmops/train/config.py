import torch
from typing import Optional , Dict 

class TrainingConfig:

  strategy : str = "ddp" #can be -> [ddp , fsdp , tp , pp , hybrid]


  #now fsdp specific configs
  fsdp_sharding_strategy: str = "full_shard"
  fsdp_cpu_offload: bool = False
  fsdp_backward_prefetch: bool = True
  fsdp_auto_wrap_min_params: float = 1e8  #wrap layers with params > 100million
  fsdp_activation_checkpointing: bool = True


  #now TP specific
  tp_size : int = 1 #tp world size
  tp_plan : Optional[Dict] = None #custom tp plan for speicific layers


  #PP specific
  pp_size : int = 1 #lmaooo
  pp_num_microbatches : int = 4


  #mixed precision
  use_mixed_precision: bool = True
  mixed_precision_dtype: torch.dtype = torch.bfloat16

  # Training hyperparameters
  batch_size: int = 32
  gradient_accumulation_steps: int = 1
  max_grad_norm: float = 1.0
  learning_rate: float = 3e-4
  weight_decay: float = 0.01
  warmup_steps: int = 1000
  max_steps: int = 100000

  #checkpointing
  checkpoint_dir: str = "./checkpoints"
  checkpoint_every_n_steps: int = 1000
  keep_last_n_checkpoints: int = 3


  #compilation
  use_torch_compile: bool = False
  compile_mode: str = "default" # can be [default , reduce-overhead , max-autotune]

