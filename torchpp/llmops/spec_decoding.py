import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional , Tuple , List


class SpeculativeDecoder:

  def __init__(
      self,
      draft_model : nn.Module,
      target_model : nn.Module,
      k : int = 4,
      temp : float = 1.0,
      top_p : float = 0.9
  ):
    
    self.draft_model = draft_model
    self.target_model = target_model
    self.k = k
    self.temp = temp
    self.top_p = top_p

    #ready for inference
    self.draft_model.eval()
    self.target_model.eval()


  @torch.no_grad()
  def sample(self, logits : torch.Tensor) -> torch.Tensor:
    
    #scale
    logits = logits / self.temp

    if self.top_p < 1.0:
      sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
      cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
     
      #remove tokens with cumulative probability above threshold
      sorted_indices_to_remove = cum_probs > self.top_p
      
      #keep at least one token
      sorted_indices_to_remove[..., 0] = False
      
      #scatter back to original indexing
      indices_to_remove = sorted_indices_to_remove.scatter(
          -1, sorted_indices, sorted_indices_to_remove
      )
      
      
      logits[indices_to_remove] = float('-inf')

    # Sample from the distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
  

  @torch.no_grad()
  def speculative_decode_step(
    self,
    input_ids : torch.Tensor,
    max_new_token : Optional[int] = None,
    pad_token_id : Optional[int] = None,
    eos_token_id : Optional[int] = None
  ) -> Tuple[torch.Tensor , dict]:
    
    device = input_ids.device
    batch_size = input_ids.shape[0]

    generated = input_ids.clone()
    total_drafted = 0
    total_accepted = 0


    while generated.shape[1] < input_ids.shape[1] + max_new_token: # type: ignore

      draft_tokens = []
      draft_input = generated.clone()

      for _ in range(self.k):

        draft_logits = self.draft_model(draft_input).logits

        draft_next_logits = draft_logits[: , -1 , :]

        next_token = self.sample(draft_next_logits)
        draft_tokens.append(next_token)


        draft_input = torch.cat([draft_input , next_token] , dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
          break


      if not draft_tokens:
        break


      draft_tokens = torch.cat(draft_tokens ,dim=1)
      total_drafted += draft_tokens.shape[1]


      verify_input = torch.cat([generated , draft_tokens] , dim=1)

      target_logits = self.target_model(verify_input).logits

      verify_positions = list(range(generated.shape[1] -1 , verify_input.shape[1] - 1))

      target_verify_logits = target_logits[: , verify_positions , :]


      target_probs = F.softmax(target_verify_logits / self.temp, dim=-1)
            
      # Get draft model probabilities for comparison
      draft_logits_verify = self.draft_model(verify_input[:, :-1]).logits
      draft_verify_logits = draft_logits_verify[:, verify_positions, :]
      draft_probs = F.softmax(draft_verify_logits / self.temp, dim=-1)
      
      # Step 3: Accept/reject tokens
      accepted_tokens = []
      for i in range(draft_tokens.shape[1]):
          draft_token = draft_tokens[:, i:i+1]
          
          # Get probabilities for the drafted token
          p_target = target_probs[:, i, :].gather(1, draft_token)
          p_draft = draft_probs[:, i, :].gather(1, draft_token)
          
          # Acceptance probability: min(1, p_target / p_draft)
          accept_prob = torch.min(torch.ones_like(p_target), p_target / (p_draft + 1e-10))
          
          # Decide acceptance
          random_val = torch.rand_like(accept_prob)
          accepted = random_val < accept_prob
          
          if accepted.all():
              accepted_tokens.append(draft_token)
              total_accepted += 1
          else:
              # p'(x) = max(0, p_target(x) - p_draft(x)) / sum(max(0, p_target - p_draft))
              adjusted_probs = torch.clamp(
                  target_probs[:, i, :] - draft_probs[:, i, :], 
                  min=0.0
              )
              adjusted_probs = adjusted_probs / (adjusted_probs.sum(dim=-1, keepdim=True) + 1e-10)
              
              resample_token = torch.multinomial(adjusted_probs, num_samples=1)
              accepted_tokens.append(resample_token)
              total_accepted += 1
              break  # Stop after first rejection
      
      if not accepted_tokens:
          # If no tokens accepted, sample from target at current position
          target_last_logits = target_logits[:, generated.shape[1] - 1, :]
          next_token = self.sample(target_last_logits)
          accepted_tokens.append(next_token)
          total_accepted += 1
      
      # Append accepted tokens
      accepted_tokens = torch.cat(accepted_tokens, dim=1)
      generated = torch.cat([generated, accepted_tokens], dim=1)
      
      # Check for EOS
      if eos_token_id is not None and (accepted_tokens[:, -1] == eos_token_id).all():
          break
  
    # Calculate statistics
    acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0
    stats = {
        'acceptance_rate': acceptance_rate,
        'total_drafted': total_drafted,
        'total_accepted': total_accepted,
        'speedup': total_accepted / (total_drafted / self.k + total_accepted) if total_drafted > 0 else 1.0
    }
    
    return generated, stats





