import torch
import torch.nn as nn

from typing import Tuple

class KVCache:
  """
  KV cache management class
  """

  def __init__(
      self,
      max_seq_len : int,
      n_kv_heads : int,
      head_dim : int,
      dtype : torch.dtype = torch.float16,
      device : torch.device = torch.device("cuda")
  ):
    
    self.max_seq_len = max_seq_len
    self.n_kv_heads = n_kv_heads
    self.head_dim = head_dim

    self.dtype = dtype
    self.device = device


    self.k_cache = torch.zeros(
      size=(1 , n_kv_heads , max_seq_len , head_dim),
      dtype=self.dtype,
      device=self.device
    )

    self.v_cache = torch.zeros(
      size=(1 , n_kv_heads , max_seq_len , head_dim),
      dtype=self.dtype,
      device=self.device
    )

    self.cache_len = 0

  def update(
      self,
      k : torch.Tensor,
      v : torch.Tensor
  ) -> Tuple[torch.Tensor , torch.Tensor]:
    """
    Update KV cache with given KV values

    Args:
        k (torch.Tensor): Key tensor
        v (torch.Tensor): value Tensor

    Returns:
        Tuple[torch.Tensor , torch.Tensor]: tuple of updated Key and Value
    """
    
    batch_size , n_heads , seq_len , head_dim = k.shape

    end_pos = self.cache_len + seq_len

    self.k_cache[: , : , self.cache_len : end_pos] = k
    self.v_cache[: , : , self.cache_len : end_pos] = v

    self.cache_len = end_pos

    return (
      self.k_cache[: , : , self.cache_len].contiguous(),
      self.v_cache[: , : , self.cache_len].contiguous()
    )
    

  def reset(self):

    self.cache_len = 0
    self.k_cache.zero_()
    self.v_cache.zero_()


  
    