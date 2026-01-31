"""
Spatio-Temporal Transformer implementation

based on the paper https://arxiv.org/abs/2001.02908

input -> images or videos
image -> (b , c , h , w)
videos -> (b , c , t , h , w) | (b , t , h , w ,c)

Spatial 


"""

import torch
import torch.nn as nn

from einops import rearrange , repeat , pack , unpack
from flash_attn import flash_attn_func

from typing import Literal

from torchpp.dlops.linear import LinearNBFp16 , LinearNBBf16
from torchpp.attention import QKV
from torchpp.utils import exists

from torchpp.attention import CrossAttention
from torchpp.dlops import PositionalEmbedding
from torchpp.dlops.normalization import LayerNorm4D


#to do : replace vanilla linear with fused linear + relu
class SpatialMLP(nn.Module):

  def __init__(
      self,
      in_dimension : int,
      hidden_dimension : int,
      dtype : torch.dtype = torch.float16
  ):
    super().__init__()

    self.layer_1 = nn.Linear(
      in_features=in_dimension,
      out_features=hidden_dimension,
      bias=False,
      dtype=dtype
    )

    self.layer_2 = nn.Linear(
      in_features=hidden_dimension,
      out_features=hidden_dimension,
      bias=False,
      dtype=dtype
    )

    self.layer_3 = nn.Linear(
      in_features=hidden_dimension,
      out_features=hidden_dimension,
      bias=False,
      dtype=dtype
    )
    
    self.activation = nn.ReLU()
  
  def forward(self , x : torch.Tensor):
    x = self.layer_1(x)
    x = self.activation(x)
    x = self.layer_2(x)
    x = self.activation(x)
    x = self.layer_3(x)

    return x


class TransformerBlock(nn.Module):

  def __init__(
      self,
      hidden_dim : int,
      num_heads : int,
      intermediate_dim : int,
      hidden_dropout : float = 0.1,
      attention_dropout : float = 0.1,
      dtype : torch.dtype = torch.float16
  ):
    
    super().__init__()

    #attention part
    self.attention = CrossAttention(
        embed_dim=hidden_dim,
        cross_dim=hidden_dim,
        n_heads=num_heads,
        qknorm=True,
        dtype=dtype,
    )
    self.attn_dropout = nn.Dropout(attention_dropout)
    self.attn_layernorm = nn.LayerNorm(hidden_dim , eps=1e-6 ,dtype=dtype)

    self.intermediate_ffn = nn.Linear(hidden_dim , intermediate_dim , dtype=dtype)
    self.output_ffn = nn.Linear(intermediate_dim , hidden_dim , dtype=dtype)
    self.ffn_dropout = nn.Dropout(hidden_dropout)
    self.ffn_layernorm = nn.LayerNorm(hidden_dim , eps=1e-6 ,dtype=dtype)
    

  def forward(self , x : torch.Tensor):
    
    #attention block
    residual = x
    x = self.attention(x , x)
    x = self.attn_dropout(x)
    x = self.attn_layernorm(x + residual)

    #ffn block
    residual = x
    x = nn.functional.relu(self.intermediate_ffn(x))
    x = self.output_ffn(x)
    x = self.ffn_dropout(x)
    x = self.ffn_layernorm(x + residual)

    return x

class SpatialTransformer(nn.Module):

  def __init__(
      self,
      kernel_size : int,
      in_channel : int,
      out_channel : int,
      num_routes : int,
      hidden_dim : int,
      num_heads : int,
      intermediate_dim : int,
      dropout : float = 0.1,
  ):
    
    super().__init__()


    self.out_channel = out_channel
    self.num_routes = num_routes
    self.kernel_size = kernel_size

    self.input_projection = None
    if(in_channel != out_channel):
      self.input_projection = nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=1
      )

    self.graph_weight = nn.Parameter(torch.empty(kernel_size * in_channel , out_channel))
    self.graph_bias = nn.Parameter(torch.zeros(out_channel))

    self.transformer_block = TransformerBlock(
      hidden_dim=out_channel,
      num_heads=num_heads,
      intermediate_dim=intermediate_dim,
      hidden_dropout=dropout,
      attention_dropout=dropout,
      dtype=torch.float16
    )

    nn.init.xavier_uniform_(self.graph_weight)


  def _gconv(
      self,
      x : torch.Tensor, # [bs , num_routes , in_channel]
      graph_kernel : torch.Tensor # [num_routes , kernel_size * num_routes]
  ):
    
    bs , n , c_in = x.shape

    #x -> [bs * c_in , num_routes]
    x_temp = x.permute(0 , 2 , 1).reshape(-1 , n)

    # x_mul = x_temp @ kernel -> [bs * c_in , kernel_size * num_routes]
    x_mul = torch.matmul(x_temp , graph_kernel)

    # reshape to [bs , c_in , ks ,n_routes]
    x_mul = x_mul.view(bs , c_in , self.kernel_size , n)

    # x_kernel -> [bs , n_routes , c_in , ks] -> [bs * n_routes , c_in * ks]
    x_ker = x_mul.permute(0 , 3 , 1 , 2).reshape(-1 , c_in * self.kernel_size)

    x_gconv = torch.matmul(x_ker , self.graph_weight) + self.graph_bias
    x_gconv = x_gconv.view(bs , n , -1)

    return x_gconv
  

  def forward(
      self,
      x : torch.Tensor ,
      graph_kernel : torch.Tensor
  ):
    """
    Args:
        x (torch.Tensor): shape -> [bs , time_step , n_route , in_channel]
        graph_kernel (torch.Tensor): shape -> [bs , time_step , n_route , out_channel]
    """

    B , T , n , C = x.shape

    if self.input_projection is not None:
      x_input = self.input_projection(x.permute(0 , 3 , 1 , 2)).permute(0 , 2 , 3 , 1) # [bs , time_step , n_route , out_channel]

    else:
      x_input = x

    x_flat = x.reshape(-1 , n , C)

    x_gconv = self._gconv(x_flat , graph_kernel)
    x_gconv = x_gconv.view(B , T , n , self.out_channel)

    x_trans = x.reshape(-1 , n , C)
    x_trans = self.transformer_block(x_trans).view(B , T , n , self.out_channel)

    output = nn.functional.relu(x_trans + x_input + 0.1 * x_gconv)

    return output


class TemporalTransformer(nn.Module):

  def __init__(
      self,
      in_channel : int,
      out_channel : int,
      num_routes : int,
      hidden_dim : int,
      num_heads : int,
      intermediate_dim : int,
      hidden_dropout : float = 0.1,
      attention_dropout : float = 0.1,
      dtype : torch.dtype = torch.float16
  ):
    
    super().__init__()

    self.out_channel = out_channel
    self.num_routes = num_routes

    self.input_projection = None
    if(in_channel != out_channel):
      self.input_projection = nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=1
      )

    self.transformer_block = TransformerBlock(
      hidden_dim=out_channel,
      num_heads=num_heads,
      intermediate_dim=intermediate_dim,
      hidden_dropout=hidden_dropout,
      attention_dropout=attention_dropout,
      dtype=dtype
    )
  
  def forward(
      self,
      x : torch.Tensor
  ):
    """
    Args:
        x (torch.Tensor): shape -> [bs , time_step , n_route , hidden_dim]
    """

    B , T , N , C = x.shape

    if self.input_projection is not None:
      x_input = self.input_projection(x.permute(0 , 3 , 1 , 2)).permute(0 , 2 , 3 , 1) # [bs , time_step , n_route , out_channel]

    elif C < self.out_channel:
      x_input = nn.functional.pad(x, (0, self.out_channel - C)) 
    
    else:
      x_input = x


    x_trans = x.permute(0 , 2 , 1 , 3).reshape(-1 , T , C) # [bs * n_route , time_step , hidden_dim]
    x_trans = self.transformer_block(x_trans).view(B , N , T , self.out_channel).permute(0 , 2 , 1 , 3)
    output = nn.functional.relu(x_trans + x_input)
    return output
  



class SpatioTemporalTransformerBlock(nn.Module):

  def __init__(
      self,
      kernel_size : int,
      in_channel : int,
      out_channel : int,
      num_routes : int,
      n_hist : int, #historical time steps
      spatial_hidden_dim : int,
      spatial_num_heads : int,
      spatial_intermediate_dim : int,
      temporal_hidden_dim : int,
      temporal_num_heads : int,
      temporal_intermediate_dim : int,
      dropout : float = 0.1,
  ):
    
    super().__init__()

    self.num_routes = num_routes
    self.n_hist = n_hist

    self.spatial_pos_embed = PositionalEmbedding(
      seq_len=num_routes,
      embed_dim=in_channel,
      dropout=dropout
    )

    self.spatial_temporal_pos_embed = PositionalEmbedding(
      seq_len=n_hist,
      embed_dim=in_channel + num_routes,
      dropout=dropout
    )

    self.temp_pos_embed = PositionalEmbedding(
      seq_len=n_hist,
      embed_dim=out_channel,
      dropout=dropout
    )

    self.spatial_transformer = SpatialTransformer(
      kernel_size=kernel_size,
      in_channel=in_channel,
      out_channel=out_channel,
      num_routes=num_routes,
      hidden_dim=spatial_hidden_dim,
      num_heads=spatial_num_heads,
      intermediate_dim=spatial_intermediate_dim,
      dropout=dropout
    )

    self.temporal_transformer = TemporalTransformer(
      in_channel=out_channel,
      out_channel=out_channel,
      num_routes=num_routes,
      hidden_dim=temporal_hidden_dim,
      num_heads=temporal_num_heads,
      intermediate_dim=temporal_intermediate_dim,
      hidden_dropout=dropout,
      attention_dropout=dropout,
      dtype=torch.float16
    )

    self.layer_norm = LayerNorm4D(num_features=out_channel)
  

  def forward(
      self,
      x : torch.Tensor ,
      graph_kernel : torch.Tensor
  ):
    """
    Args:
        x (torch.Tensor): shape -> [bs , time_step , n_route , in_channel]
        graph_kernel (torch.Tensor): shape -> [bs , time_step , n_route , out_channel]
    """
    B , T , N , C = x.shape


    #spatial pos embed
    x_flat = x.reshape(-1 , N , C)
    x = self.spatial_pos_embed(x_flat).view(B , T , N , -1)

    #temp pos embed
    x = x.permute(0 , 2 , 1 , 3).reshape(-1 , T , x.size(-1)) # [bs * n_route , time_step , in_channel]

    x = self.spatial_temporal_pos_embed(x)
    x = x.view(B , N , T , -1).permute(0 , 2 , 1 , 3) # [bs , time_step , n_route , in_channel + n_route]

    #spatial trasnformer
    x_spatial = self.spatial_transformer(x , graph_kernel)
    
    
    #temp pos embed
    x_spatial = x_spatial.permute(0 , 2 , 1 , 3).reshape(-1 , T , x_spatial.size(-1)) # [bs * n_route , time_step , out_channel]
    x_spatial = self.temp_pos_embed(x_spatial)
    x_spatial = x_spatial.view(B , N , T , -1).permute(0 , 2 , 1 , 3) # [bs , time_step ,

    x_o = self.temporal_transformer(x_spatial)

    x_ln = self.layer_norm(x_o)

    return x_ln