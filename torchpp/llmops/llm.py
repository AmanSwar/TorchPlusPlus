import torch
import torch.nn as nn

from typing import Optional , List , Callable
from torchpp.llmops.kvcache import KVCache

from abc import ABC , abstractmethod

class LLMInference(nn.Module , ABC):

  def __init__(self):
    super().__init__()

    self.kv_cache_array : Optional[List[KVCache]] = None

  @abstractmethod
  def forward_impl(
    self,
    input_ids : torch.Tensor,
    kv_caches : Optional[List[KVCache]] = None,
    cache_position : Optional[List] = None, 
  ) -> torch.Tensor:
    """
    Core Forward pass method

    Args:
        input_ids (torch.Tensor): Input token of Ids [bs , seq_len]
        kv_caches (Optional[List[KVCache]], optional): List of KV cahce for each transformer layer.
        cache_position (Optional[List], optional): Position of cahce for RoPE compute.

    Returns:
        logits (torch.Tensor) : output logits
    """
    pass

  @abstractmethod
  def get_num_layers(self) -> int:
    pass

  @abstractmethod
  def get_kv_heads(self) -> int:
    pass

  @abstractmethod
  def get_head_dim(self) -> int:
    pass

  @abstractmethod
  def get_dtype(self) -> torch.dtype:
    pass
  
  def setup_kv_cache(
      self,
      max_seq_len : int
  ):
    
    device = next(self.parameters()).device
    
    self.kv_cache_array = []

    for _ in range(self.get_num_layers()):

      kv_cache = KVCache(
        max_seq_len=max_seq_len,
        n_kv_heads=self.get_kv_heads(),
        head_dim=self.get_head_dim(),
        dtype=self.get_dtype(),
        device=device
      )

      self.kv_cache_array.append(kv_cache)

  def reset_kv_cache(self):

    if self.kv_cache_array:
      for cache in self.kv_cache_array:
        cache.reset()

  def clear_kv_cache(self):
    """clear KV caches and free mem"""

    if self.kv_cache_array:
        for cache in self.kv_cache_array:
            del cache.k_cache
            del cache.v_cache

        self.kv_caches = None
        torch.cuda.empty_cache()


  def forward(
      self,
      input_ids : torch.Tensor,
      use_cache : bool = False,
      cache_position : Optional[int] = None
  ) -> torch.Tensor:
    
    if use_cache and self.kv_cache_array is None:

      raise ValueError(
        "KV cache is not initialized , call setup_kv_cache() or use normal nn.Module"
      )
    

    kv_cache_array = self.kv_cache_array if use_cache else None

    return self.forward_impl(
      input_ids,
      kv_cache_array,
      cache_position
    )
    
  
  def sample_top_p(
      self,
      logits : torch.Tensor,
      top_p : float = 0.9,
      temp : float = 1.0,
      filter_value : float = -float("inf")
  ) -> torch.Tensor:
    
    #scale down by temp
    logits = logits/ temp

    if top_p < 1.0:
      
      sorted_logits , sorted_indices = torch.sort(logits , descending=True)

      cum_probs = torch.cumsum(
        torch.softmax(sorted_logits , dim=1) , dim=1
      )

      #remove the token with cum probab above top_p 
      sorted_indices_to_rm =  cum_probs > top_p
      sorted_indices_to_rm[... , 1 : ] = sorted_indices_to_rm[... : -1].clone()

      sorted_indices_to_rm[... : 0] = 0


      indices_to_rm = sorted_indices_to_rm.scatter(
        1 , sorted_indices , sorted_indices_to_rm
      )

      logits[indices_to_rm] = filter_value


    probs = torch.softmax(logits , dim=-1)
    next_token  = torch.multinomial(probs , num_samples=1)

    return next_token
  

  def sample_top_k(
      self,
      logits: torch.Tensor,
      top_k: int = 50,
      temp: float = 1.0,
      filter_value: float = -float("inf"),
  ) -> torch.Tensor:
      #scale down logits
      logits = logits / temp

      #take out top k
      top_k = min(top_k, logits.size(-1))

      values, indices = torch.topk(logits, top_k)

      #everything else to -inf
      logits_filtered = torch.full_like(logits, filter_value)

      logits_filtered.scatter_(1, indices, values)

      #sample
      probs = torch.softmax(logits_filtered, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1)

      
      return next_token
  

  def sample_greedy(
      self,
      logits : torch.Tensor
  ):
    
    return torch.argmax(logits , dim=-1 , keepdim=True)
  
  def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        strategy: str,
    ) -> torch.Tensor:
      if strategy == "greedy":
          return self.sample_greedy(logits)
      elif strategy == "top_k" and top_k is not None:
          return self.sample_top_k(logits, top_k, temperature)
      elif strategy == "top_p":
          return self.sample_top_p(logits, top_p, temperature)
      else:
          raise ValueError(f"Unknown sampling strategy: {strategy}")
      
  def _should_stop(
      self,
      token: torch.Tensor,
      stop_token_ids: List[int],
      eos_token_id: Optional[int],
  ) -> bool:
      """Check if generation should stop"""
      token_id = token.item()
      
      # Check EOS token
      if eos_token_id is not None and token_id == eos_token_id:
          return True
      
      # Check stop tokens
      if token_id in stop_token_ids:
          return True
      
      return False
  

  @torch.inference_mode()
  def generate(
    self,
    input_ids : torch.Tensor,
    max_new_token : int = 100,
    temp : float = 1.0,
    top_p : float = 0.9,
    top_k : Optional[int] = None,
    sampling_strategy : str = "top_p",
    stop_token_ids : Optional[List[int]] = None,
    eos_token_id : Optional[int] = None,
    stream : bool = False,
    stream_callback : Optional[Callable[[torch.Tensor], None]] = None
  ):

    max_seq_len = input_ids.shape[1] + max_new_token
    self.setup_kv_cache(max_seq_len)
    self.reset_kv_cache()

    batch_size, seq_len = input_ids.shape
    stop_token_ids = stop_token_ids or []

    #PREFILL
    logits = self.forward(input_ids, use_cache=True, cache_position=0)
    next_token_logits = logits[:, -1, :]

    next_token = self._sample_token(
        next_token_logits, temp, top_p, top_k, sampling_strategy
      )

    if stream and stream_callback:
        stream_callback(next_token)

    generated_tokens = [next_token]

    if self._should_stop(next_token, stop_token_ids, eos_token_id):
      return torch.cat([input_ids] + generated_tokens, dim=1)
    
    for i in range(max_new_token - 1):
      cache_position = seq_len + i
      logits = self.forward(
          next_token, use_cache=True, cache_position=cache_position
      )
      next_token_logits = logits[:, -1, :]

      # Sample next token
      next_token = self._sample_token(
          next_token_logits, temp, top_p, top_k, sampling_strategy
      )

      if stream and stream_callback:
          stream_callback(next_token)

      generated_tokens.append(next_token)

      # Check for early stopping
      if self._should_stop(next_token, stop_token_ids, eos_token_id):
          break

    # Concatenate all tokens
    generated_sequence = torch.cat([input_ids] + generated_tokens, dim=1)
    return generated_sequence


  def get_cache_info(self) -> dict:
    """Get information about current cache state"""
    if not self.kv_caches:
        return {"initialized": False}

    return {
        "initialized": True,
        "num_layers": len(self.kv_caches),
        "cache_length": self.kv_caches[0].cache_len if self.kv_caches else 0,
        "max_seq_len": self.kv_caches[0].max_seq_len if self.kv_caches else 0,
        "n_kv_heads": self.kv_caches[0].n_kv_heads if self.kv_caches else 0,
        "head_dim": self.kv_caches[0].head_dim if self.kv_caches else 0,
    }

  def estimate_memory_usage(self, max_seq_len: int) -> dict:
    """Estimate memory usage for KV cache"""
    num_layers = self.get_num_layers()
    n_kv_heads = self.get_kv_heads()
    head_dim = self.get_head_dim()
    dtype = self.get_dtype()

    # Bytes per element
    bytes_per_element = 2 if dtype == torch.float16 else 4

    # Memory per layer (K and V)
    memory_per_layer = (
        2 * n_kv_heads * max_seq_len * head_dim * bytes_per_element
    )
    
    total_memory = num_layers * memory_per_layer

    return {
        "total_bytes": total_memory,
        "total_mb": total_memory / (1024 ** 2),
        "total_gb": total_memory / (1024 ** 3),
        "per_layer_mb": memory_per_layer / (1024 ** 2),
        "num_layers": num_layers,
    }




  

     
    