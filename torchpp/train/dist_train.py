import torch
import torch.nn as nn
#dist
import torch.distributed as dist
#ddp
from torch.nn.parallel import DistributedDataParallel as DDP
#fsdp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
    transformer_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)


#TP
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
TENSOR_PARALLEL_AVAILABLE = True

# PP
from torch.distributed.pipelining import pipeline
PIPELINE_PARALLEL_AVAILABLE = True

#utils
import functools
import os
from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from torchpp.train.config import TrainingConfig 

class ParallelStrategy(Enum):

  DDP = "ddp"
  FSDP = "fsdp"
  TENSOR_PARALLEL = "tensor_parallel"
  PIPELINE_PARALLEL = "pipeline_parallel"
  HYBRID = "hybrid"  # Combination of multiple strategies


class DistrbutedTrainer:

  def __init__(
      self,
      model : nn.Module,
      config : TrainingConfig,
      optimizer : Optional[torch.optim.Optimizer] = None,
      lr_sched : Optional[Any] = None,
      loss_function : Optional[Callable] = None
  ):
    
    self.model = model
    self.config = config
    self.optimizer = optimizer
    self.lr_scheduler = lr_sched
    self.loss_fn = loss_function or self._default_loss_fn
    
    #setup 
    self._setup_distributed()

    #wrap the model
    self._wrap_model()

    #setup the optim (if not provided)
    if self.optimizer is None:
      self._setup_optimizer()

    #training state
    self.global_step = 0
    self.current_epoch = 0
    
    #create chkpt dir
    if self.is_main_process:
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    
  def _setup_distributed(self):
    
    if not dist.is_initialized():
      dist.init_process_group(backend="nccl")

    self.rank = dist.get_rank()
    self.world_size = dist.get_world_size()
    self.local_rank = int(os.environ.get("LOCAL_RANK" , 0))

    #set device
    torch.cuda.set_device(self.local_rank)
    self.device = torch.device(f"cuda:{self.local_rank}")

    self.is_main_process = self.rank == 0


    if self.is_main_process:
      logger.info(f"initialized distrbuted trainign with {self.world_size} process")
      logger.info(f"strategy : {self.config.strategy}")


  def _wrap_model(self):
    self.model = self.model.to(self.device)

    strategy = ParallelStrategy(self.config.strategy)

    if strategy == ParallelStrategy.DDP:
        self._setup_ddp()
    elif strategy == ParallelStrategy.FSDP:
        self._setup_fsdp()
    elif strategy == ParallelStrategy.TENSOR_PARALLEL:
        self._setup_tensor_parallel()
    elif strategy == ParallelStrategy.PIPELINE_PARALLEL:
        self._setup_pipeline_parallel()
    elif strategy == ParallelStrategy.HYBRID:
        self._setup_hybrid()
    else:
        raise ValueError(f"Unknown strategy: {self.config.strategy}")


    if self.config.use_torch_compile:
      if self.is_main_process:
          logger.info("compiling model .... (keep patience don't doom scroll!!)")
      self.model = torch.compile(
          self.model,
          mode=self.config.compile_mode
      )


  def _setup_ddp(self):
     
    if self.is_main_process:
       
       logger.info("setting ddp")

    self.model = DDP(
       self.model,
       device_ids=[self.local_rank],
       output_device=self.local_rank,
       gradient_as_bucket_view=True,
       static_graph=False,
    )


  
  def _setup_fsdp(self):   
    if self.is_main_process:
      logging.info("settign fsdp!")

    if self.config.use_mixed_precision:
      mixed_precision_policy = MixedPrecision(
          param_dtype=self.config.mixed_precision_dtype,
          reduce_dtype=self.config.mixed_precision_dtype,
          buffer_dtype=self.config.mixed_precision_dtype,
      )
    else:
      mixed_precision_policy = None

    sharding_strategy_map = {
      "full_shard": ShardingStrategy.FULL_SHARD,
      "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
      "no_shard": ShardingStrategy.NO_SHARD,
      "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    sharding_strategy = sharding_strategy_map[self.config.fsdp_sharding_strategy]

    auto_wrap_policy = functools.partial(
      size_based_auto_wrap_policy,
      min_num_params=int(self.config.fsdp_auto_wrap_min_params),
    )

    cpu_offload = CPUOffload(offload_params=True) if self.config.fsdp_cpu_offload else None

    #backward prefetch
    backward_prefetch = (
      BackwardPrefetch.BACKWARD_PRE if self.config.fsdp_backward_prefetch else None
    )

    self.model = FSDP(
      self.model,
      sharding_strategy=sharding_strategy,
      auto_wrap_policy=auto_wrap_policy,
      mixed_precision=mixed_precision_policy,
      backward_prefetch=backward_prefetch,
      cpu_offload=cpu_offload,
      device_id=torch.cuda.current_device(),
      limit_all_gathers=True,
      use_orig_params=True, 
    )

    if self.config.fsdp_activation_checkpointing:
      self._apply_activation_checkpointing()

  def _apply_activation_checkpointing(self):
     
    if self.is_main_process:
      logger.info("applyuign activation checkpointing")

    def check_fn(module):

      module_name = module.__class__.__name__
      return "transformer" in module_name or "block" in module_name
    
    non_reentrant_wrapper = functools.partial(
      checkpoint_wrapper,
      checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    
    apply_activation_checkpointing(
      self.model,
      checkpoint_wrapper_fn=non_reentrant_wrapper,
      check_fn=check_fn,
    )
  
  def _setup_tensor_parallel(self):
    if not TENSOR_PARALLEL_AVAILABLE:
        raise ImportError("Tensor Parallelism requires PyTorch >= 2.0")
    
    if self.is_main_process:
        logger.info(f"Setting up Tensor Parallelism (TP size: {self.config.tp_size})...")
    
    # create tensor parallel device mesh
    from torch.distributed.device_mesh import init_device_mesh
    
    device_mesh = init_device_mesh("cuda", (self.config.tp_size,))
    
    # This is model-specific - need to specify which layers to parallelize
    tp_plan = self.config.tp_plan or self._get_default_tp_plan()
    
    # Apply tensor parallelism
    self.model = parallelize_module(
        self.model,
        device_mesh,
        tp_plan,
    )
  

  def _get_default_tp_plan(self) -> Dict:
    """generic plan if no plan is provided """
    return {
        "attn.Wq": ColwiseParallel(),
        "attn.Wk": ColwiseParallel(),
        "attn.Wv": ColwiseParallel(),
        "attn.out_projection": RowwiseParallel(),
        
        "ffn.linear_layer1": ColwiseParallel(),
        "ffn.linear_layer2": RowwiseParallel(),
    }
  

  def _setup_pipeline_parallel(self):
    """Setup Pipeline Parallel"""
    if not PIPELINE_PARALLEL_AVAILABLE:
        raise ImportError("Pipeline Parallelism requires PyTorch >= 2.1")
    
    if self.is_main_process:
        logger.info(f"Setting up Pipeline Parallelism (PP size: {self.config.pp_size})...")
    
    raise NotImplementedError(
        "Pipeline parallelism requires model-specific stage splitting. "
        "Please implement _split_model_into_stages() for your model."
    )
  

  def _setup_hybrid(self):
    """hybrid paralle -> (TP + DP or PP + DP)"""
    
    if self.is_main_process:
        logger.info("Setting up Hybrid Parallelism...")
    
    from torch.distributed.device_mesh import init_device_mesh
    
    dp_size = self.world_size // self.config.tp_size
    device_mesh = init_device_mesh(
        "cuda",
        (dp_size, self.config.tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    
    tp_plan = self.config.tp_plan or self._get_default_tp_plan()
    self.model = parallelize_module(
        self.model,
        device_mesh["tp"],
        tp_plan,
    )
    
    self.model = FSDP(
        self.model,
        device_mesh=device_mesh["dp"],
        use_orig_params=True,
    )

  def _setup_optimizer(self):
    
    decay_params = []
    no_decay_params = []

    for name, param in self.model.named_parameters():
      if param.requires_grad:
        if "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {
          "params": decay_params,
          "weight_decay": self.config.weight_decay,
        },
        {
          "params": no_decay_params,
          "weight_decay": 0.0,
        },
    ]
    
    self.optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=self.config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    
    if self.is_main_process:
        logger.info(f"Initialized AdamW optimizer with lr={self.config.learning_rate}")
    

  def _default_loss_fn(self, logits, labels):
    """default to cross entropy loss if no loss provided"""
    return nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
  
  def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """single training step"""
    self.model.train()
    
    input_ids = batch["input_ids"].to(self.device)
    labels = batch["labels"].to(self.device)
    
    if self.config.use_mixed_precision:
        with torch.autocast(device_type="cuda", dtype=self.config.mixed_precision_dtype):
            logits = self.model(input_ids)
            loss = self.loss_fn(logits, labels)
    else:
        logits = self.model(input_ids)
        loss = self.loss_fn(logits, labels)
    
    loss = loss / self.config.gradient_accumulation_steps
    
    loss.backward()
    
    return {"loss": loss.item() * self.config.gradient_accumulation_steps}
  

  def train(
    self,
    train_dataloader,
    eval_dataloader=None,
    num_epochs: Optional[int] = None,
  ):
    if self.is_main_process:
      logger.info("Starting training...")
      logger.info(f"Total steps: {self.config.max_steps}")
      logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
    
    num_epochs = num_epochs or (self.config.max_steps // len(train_dataloader)) + 1
    
    for epoch in range(num_epochs):
      self.current_epoch = epoch
      
      if self.is_main_process:
          logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
      
      for step, batch in enumerate(train_dataloader):
          #training step
          metrics = self.train_step(batch)
          
          # grad accumulation
          if (step + 1) % self.config.gradient_accumulation_steps == 0:
              # clip gradients
              if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
              
              self.optimizer.step() # type: ignore
              
              if self.lr_scheduler is not None:
                  self.lr_scheduler.step()
              
              self.optimizer.zero_grad() # type: ignore
              self.global_step += 1
              
              if self.global_step % self.config.log_every_n_steps == 0: # type: ignore
                  self._log_metrics(metrics)
              
              if self.global_step % self.config.checkpoint_every_n_steps == 0:
                  self._save_checkpoint()
              
              if self.global_step >= self.config.max_steps:
                  if self.is_main_process:
                      logger.info(f"Reached max steps ({self.config.max_steps})")
                  return
      
      if eval_dataloader is not None:
          eval_metrics = self.evaluate(eval_dataloader)
          if self.is_main_process:
              logger.info(f"Eval metrics: {eval_metrics}")
    
    if self.is_main_process:
        logger.info("Training completed!")

  def evaluate(self, eval_dataloader) -> Dict[str, float]:
    """Evaluation loop"""
    self.model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
      for batch in eval_dataloader:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        if self.config.use_mixed_precision:
          with torch.autocast(device_type="cuda", dtype=self.config.mixed_precision_dtype):
              logits = self.model(input_ids)
              loss = self.loss_fn(logits, labels)

        else:
          logits = self.model(input_ids)
          loss = self.loss_fn(logits, labels)
          
          total_loss += loss.item()
          num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    loss_tensor = torch.tensor([avg_loss], device=self.device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    
    return {"eval_loss": loss_tensor.item()}


  def _log_metrics(self, metrics: Dict[str, float]):
    """Log training metrics""" #given by claude
    if self.is_main_process:
        lr = self.optimizer.param_groups[0]["lr"] # type: ignore
        log_str = f"Step {self.global_step} | "
        log_str += f"Loss: {metrics['loss']:.4f} | "
        log_str += f"LR: {lr:.2e}"
        logger.info(log_str) 

  def _save_checkpoint(self):
    if not self.is_main_process:
        return
    
    checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint-{self.global_step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    if isinstance(self.model, (DDP, FSDP)):
        state_dict = self.model.state_dict()
    else:
        state_dict = self.model.state_dict()
    
    checkpoint = {
        "model": state_dict,
        "optimizer": self.optimizer.state_dict(), # type: ignore
        "global_step": self.global_step,
        "epoch": self.current_epoch,
        "config": self.config.__dict__,
    }
    
    if self.lr_scheduler is not None:
        checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path / "checkpoint.pt")
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    self._cleanup_old_checkpoints()


  def _cleanup_old_checkpoints(self):
    """Remove old checkpoints, keeping only the last N"""
    checkpoint_dir = Path(self.config.checkpoint_dir)
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda x: int(x.name.split("-")[1]),
    )
    
    if len(checkpoints) > self.config.keep_last_n_checkpoints:
      for checkpoint in checkpoints[:-self.config.keep_last_n_checkpoints]:
        import shutil
        shutil.rmtree(checkpoint)
        logger.info(f"Removed old checkpoint: {checkpoint}")

  def load_checkpoint(self, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=self.device)
    
    if isinstance(self.model, (DDP, FSDP)):
      self.model.load_state_dict(checkpoint["model"])
    else:
      self.model.load_state_dict(checkpoint["model"])
    
    self.optimizer.load_state_dict(checkpoint["optimizer"]) # type: ignore
    
    self.global_step = checkpoint["global_step"]
    self.current_epoch = checkpoint["epoch"]
    
    if self.lr_scheduler is not None and "lr_scheduler" in checkpoint:
      self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    
    if self.is_main_process:
      logger.info(f"Loaded checkpoint from {checkpoint_path}")
      logger.info(f"Resuming from step {self.global_step}")