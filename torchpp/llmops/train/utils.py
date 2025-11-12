#This whole code was given by claude 4.5 sonnet (idk why)

import torch
import torch.distributed as dist
from typing import Dict, Tuple, Optional , Any
from dataclasses import dataclass
import subprocess
import os


@dataclass
class ModelStats:
  """Model statistics for strategy selection"""
  total_params: int
  total_params_billions: float
  layer_params: int
  embed_params: int
  lm_head_params: int



def calculate_model_stats(model: torch.nn.Module) -> ModelStats:
  """Calculate model statistics"""
  total_params = sum(p.numel() for p in model.parameters())
  
  # Try to get layer-specific stats
  layer_params = 0
  if hasattr(model, 'transformer_blocs') or hasattr(model, 'layers'):
      layers = getattr(model, 'transformer_blocs', getattr(model, 'layers', []))
      if len(layers) > 0:
          layer_params = sum(p.numel() for p in layers[0].parameters())
  
  embed_params = 0
  if hasattr(model, 'tok_embed') or hasattr(model, 'token_embedding'):
      embed = getattr(model, 'tok_embed', getattr(model, 'token_embedding', None))
      if embed:
          embed_params = sum(p.numel() for p in embed.parameters())
  
  lm_head_params = 0
  if hasattr(model, 'out_head') or hasattr(model, 'lm_head'):
      lm_head = getattr(model, 'out_head', getattr(model, 'lm_head', None))
      if lm_head:
          lm_head_params = sum(p.numel() for p in lm_head.parameters())
  
  return ModelStats(
      total_params=total_params,
      total_params_billions=total_params / 1e9,
      layer_params=layer_params,
      embed_params=embed_params,
      lm_head_params=lm_head_params,
  )

def recommend_strategy(
    model: torch.nn.Module,
    num_gpus: int,
    gpu_memory_gb: int = 80,  # A100
    batch_size: int = 1,
) -> Dict[str, Any]:
  """
  Recommend training strategy based on model size and available resources.
  
  Returns:
      Dictionary with recommendations and explanations
  """
  stats = calculate_model_stats(model)
  
  # Estimate memory requirements (rough approximation)
  # Model params (FP16) + Gradients + Optimizer states (Adam: 2x params)
  # + Activations (depends on batch size and sequence length)
  param_memory_gb = (stats.total_params * 2) / 1e9  # FP16
  gradient_memory_gb = param_memory_gb
  optimizer_memory_gb = param_memory_gb * 2  # Adam states
  activation_memory_gb = batch_size * 2  # Rough estimate
  
  total_memory_gb = (
      param_memory_gb + gradient_memory_gb + optimizer_memory_gb + activation_memory_gb
  )
  
  recommendations = {
      "model_stats": stats,
      "estimated_memory_gb": total_memory_gb,
      "memory_per_gpu_gb": gpu_memory_gb,
  }
  
  # Decision tree for strategy selection
  if stats.total_params_billions < 1.0 and total_memory_gb < gpu_memory_gb:
      # Small model that fits on one GPU
      recommendations["strategy"] = "ddp"
      recommendations["reason"] = (
          "Model is small enough to fit on each GPU. "
          "DDP is simplest and has lowest communication overhead."
      )
      recommendations["config"] = {
          "strategy": "ddp",
          "use_mixed_precision": True,
          "gradient_accumulation_steps": max(1, 32 // batch_size),
      }
  
  elif stats.total_params_billions < 7.0:
      # Medium model (1B-7B params)
      recommendations["strategy"] = "fsdp"
      recommendations["reason"] = (
          "Model requires parameter sharding. "
          "FSDP with FULL_SHARD provides good balance of memory and speed."
      )
      recommendations["config"] = {
          "strategy": "fsdp",
          "fsdp_sharding_strategy": "full_shard",
          "fsdp_activation_checkpointing": True,
          "use_mixed_precision": True,
          "gradient_accumulation_steps": max(1, 32 // (batch_size * num_gpus)),
      }
  
  elif stats.total_params_billions < 30.0:
      # Large model (7B-30B params)
      if num_gpus >= 8:
          recommendations["strategy"] = "hybrid"
          recommendations["reason"] = (
              "Large model benefits from hybrid parallelism. "
              "Use 2-4 way TP within nodes, FSDP across nodes."
          )
          recommendations["config"] = {
              "strategy": "hybrid",
              "tp_size": min(4, num_gpus // 2),
              "fsdp_sharding_strategy": "full_shard",
              "fsdp_activation_checkpointing": True,
              "fsdp_cpu_offload": False,
              "use_mixed_precision": True,
              "mixed_precision_dtype": "bfloat16",
              "gradient_accumulation_steps": 32,
          }
      else:
          recommendations["strategy"] = "fsdp"
          recommendations["reason"] = (
              "Use FSDP with CPU offload if OOM. "
              "Consider activation checkpointing."
          )
          recommendations["config"] = {
              "strategy": "fsdp",
              "fsdp_sharding_strategy": "full_shard",
              "fsdp_activation_checkpointing": True,
              "fsdp_cpu_offload": total_memory_gb > (num_gpus * gpu_memory_gb * 0.8),
              "use_mixed_precision": True,
              "gradient_accumulation_steps": 64,
          }
  
  else:
      # Very large model (>30B params)
      recommendations["strategy"] = "hybrid"
      recommendations["reason"] = (
          "Very large model requires aggressive parallelism. "
          "Use 4-8 way TP + FSDP + activation checkpointing + CPU offload."
      )
      recommendations["config"] = {
          "strategy": "hybrid",
          "tp_size": min(8, num_gpus),
          "fsdp_sharding_strategy": "full_shard",
          "fsdp_activation_checkpointing": True,
          "fsdp_cpu_offload": True,
          "use_mixed_precision": True,
          "mixed_precision_dtype": "bfloat16",
          "gradient_accumulation_steps": 128,
      }
  
  return recommendations

def estimate_training_memory(
  num_params: int,
  batch_size: int,
  seq_length: int,
  hidden_size: int,
  num_layers: int,
  use_mixed_precision: bool = True,
  use_gradient_checkpointing: bool = False,
  sharding_strategy: str = "full_shard",
) -> Dict[str, float]:
  """
  Estimate memory requirements for training.
  
  Args:
      num_params: Total model parameters
      batch_size: Batch size per GPU
      seq_length: Sequence length
      hidden_size: Hidden dimension
      num_layers: Number of transformer layers
      use_mixed_precision: Using FP16/BF16
      use_gradient_checkpointing: Using activation checkpointing
      sharding_strategy: FSDP sharding strategy
      
  Returns:
      Dictionary with memory breakdown in GB
  """
  bytes_per_param = 2 if use_mixed_precision else 4
  
  # Model parameters
  param_memory = num_params * bytes_per_param / 1e9
  
  # Gradients
  gradient_memory = param_memory
  
  # Optimizer states (Adam: 2x params for momentum and variance)
  optimizer_memory = num_params * 2 * 4 / 1e9  # Always FP32
  
  # Activations (rough estimate)
  # Per layer: batch_size * seq_length * hidden_size * 4 (Q, K, V, O)
  if use_gradient_checkpointing:
      # Only store activations for 1 layer at a time
      activation_memory = (
          batch_size * seq_length * hidden_size * 4 * bytes_per_param / 1e9
      )
  else:
      # Store all activations
      activation_memory = (
          num_layers * batch_size * seq_length * hidden_size * 4 * bytes_per_param / 1e9
      )
  
  # Apply FSDP sharding factor
  sharding_factor = 1  # Placeholder - actual depends on world size
  if sharding_strategy == "full_shard":
      # Shard params, grads, and optimizer states
      param_memory = param_memory / sharding_factor
      gradient_memory = gradient_memory / sharding_factor
      optimizer_memory = optimizer_memory / sharding_factor
  elif sharding_strategy == "shard_grad_op":
      # Only shard grads and optimizer
      gradient_memory = gradient_memory / sharding_factor
      optimizer_memory = optimizer_memory / sharding_factor
  
  total_memory = (
      param_memory + gradient_memory + optimizer_memory + activation_memory
  )
  
  return {
      "param_memory_gb": param_memory,
      "gradient_memory_gb": gradient_memory,
      "optimizer_memory_gb": optimizer_memory,
      "activation_memory_gb": activation_memory,
      "total_memory_gb": total_memory,
      "peak_memory_gb": total_memory * 1.2,  # Add 20% overhead
  }

class TrainingMonitor:
  """Monitor GPU usage and training metrics"""
  
  @staticmethod
  def get_gpu_memory_info() -> Dict[int, Dict[str, float]]:
    """Get memory usage for all GPUs"""
    if not torch.cuda.is_available():
        return {}
    
    gpu_info = {}
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1e9
        memory_reserved = torch.cuda.memory_reserved(i) / 1e9
        max_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        
        gpu_info[i] = {
            "allocated_gb": memory_allocated,
            "reserved_gb": memory_reserved,
            "total_gb": max_memory,
            "utilization_pct": (memory_allocated / max_memory) * 100,
        }
    
    return gpu_info
  
  @staticmethod
  def print_gpu_memory():
    """Print GPU memory usage"""
    gpu_info = TrainingMonitor.get_gpu_memory_info()
    
    print("\n" + "="*60)
    print("GPU Memory Usage:")
    print("="*60)
    
    for gpu_id, info in gpu_info.items():
        print(f"GPU {gpu_id}:")
        print(f"  Allocated: {info['allocated_gb']:.2f} GB")
        print(f"  Reserved:  {info['reserved_gb']:.2f} GB")
        print(f"  Total:     {info['total_gb']:.2f} GB")
        print(f"  Usage:     {info['utilization_pct']:.1f}%")
    
    print("="*60 + "\n")
  
  @staticmethod
  def log_throughput(
    global_step: int,
    start_time: float,
    tokens_processed: int,
  ):
    """Log training throughput"""
    import time
    elapsed = time.time() - start_time
    tokens_per_sec = tokens_processed / elapsed
    
    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"Step {global_step} | "
              f"Tokens/sec: {tokens_per_sec:.0f} | "
              f"Time: {elapsed:.2f}s")