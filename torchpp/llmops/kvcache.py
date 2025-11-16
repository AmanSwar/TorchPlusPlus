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

    #DON'T PREALLOCATE DURING INFERENCE
    #we will allocate on first use
    self.k_cache : None | torch.Tensor = None
    self.v_cache : None | torch.Tensor = None 

    self.cache_len = 0

  def _ensure_allocated(self , length : int):
    #will only allocate once on first use

    if self.k_cache is None:
      alloc_len = min(length , self.max_seq_len)
      self.k_cache = torch.zeros(
        (1 , self.n_kv_heads , alloc_len , self.head_dim),
        dtype = self.dtype,
        device = self.device
      )
      self.v_cache = torch.zeros(
        (1 , self.n_kv_heads , alloc_len , self.head_dim),
        dtype = self.dtype,
        device = self.device
      )
    
  def update(
      self,
      k : torch.Tensor,
      v : torch.Tensor,
      index : int |None = None
  ) -> Tuple[torch.Tensor , torch.Tensor]:
    """
    Update KV cache with given KV values

    Args:
        k (torch.Tensor): Key tensor
        v (torch.Tensor): value Tensor

    Returns:
        Tuple[torch.Tensor , torch.Tensor]: tuple of updated Key and Value
    """
    self._ensure_allocated(1)
    
    
    batch_size , n_heads , seq_len , head_dim = k.shape
    if index is None:
      end_pos = self.cache_len + seq_len

      self.k_cache[: , : , self.cache_len : end_pos] = k
      self.v_cache[: , : , self.cache_len : end_pos] = v

      self.cache_len = end_pos

      return (
        self.k_cache[: , : , self.cache_len].contiguous(),
        self.v_cache[: , : , self.cache_len].contiguous()
      )
    
    else:
      if index >= seq_len:
        #expand the kv cache
        new_size = min(index + 100 , self.max_seq_len)
        new_k_cache = torch.zeros(
          (1 , self.n_kv_heads , new_size , self.head_dim),
          dtype = self.dtype,
          device = self.device
        )
        new_v_cache = torch.zeros(
          (1 , self.n_kv_heads , new_size , self.head_dim),
          dtype = self.dtype,
          device = self.device
        )
        new_k_cache[: , : , :seq_len , :] = self.k_cache
        new_v_cache[: , : , :seq_len , :] = self.v_cache
        self.k_cache = new_k_cache
        self.v_cache = new_v_cache
      self.k_cache[: , : , index , :] = k.squeeze(2)
      self.v_cache[: , : , index , :] = v.squeeze(2)
      
      self.cache_len = max(self.cache_len , index + 1)
      return (
        self.k_cache[: , : , :self.cache_len , :].contiguous(),
        self.v_cache[: , : , :self.cache_len , :].contiguous()
      )

      

  def reset(self):

    self.cache_len = 0
    self.k_cache.zero_()
    self.v_cache.zero_()


  
    