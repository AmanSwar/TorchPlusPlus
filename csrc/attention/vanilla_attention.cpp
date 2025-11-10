#include <stdexcept>
#include <torch/extension.h>

#include "ATen/core/TensorBody.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "flash_attn.cuh" // conttains the kernel launch function

namespace torchpp{

torch::Tensor vanillaFlashAttention(
  torch::Tensor Q, // B , H , N , D
  torch::Tensor K,
  torch::Tensor V
  // torch::Tensor O
){

  

  const int QKV_batch = Q.size(0);
  const int QKV_head = Q.size(1);
  const int QKV_seqlen = Q.size(2);

  const int headDim = Q.size(3);

  auto O = torch::empty(
    {Q.size(0) , Q.size(1) , Q.size(2) , Q.size(3)},
    torch::TensorOptions().dtype(torch::kHalf).device(Q.device())
  );


  switch(headDim){

    case 32:
      launch_flash_attn<32 , 2>(
      reinterpret_cast<half*>(Q.data_ptr()),
      reinterpret_cast<half* >(K.data_ptr()),
      reinterpret_cast<half* >(V.data_ptr()),
      reinterpret_cast<half* >(O.data_ptr()),
      QKV_batch ,QKV_head , QKV_seqlen
    );
      break;

    case 64:
      launch_flash_attn<64,2>(
        reinterpret_cast<half*>(Q.data_ptr()),
        reinterpret_cast<half* >(K.data_ptr()),
        reinterpret_cast<half* >(V.data_ptr()),
        reinterpret_cast<half* >(O.data_ptr()),
        QKV_batch ,QKV_head , QKV_seqlen
      );
      break;


    case 96:
      launch_flash_attn<96 , 2>(
        reinterpret_cast<half*>(Q.data_ptr()),
        reinterpret_cast<half* >(K.data_ptr()),
        reinterpret_cast<half* >(V.data_ptr()),
        reinterpret_cast<half* >(O.data_ptr()),
        QKV_batch ,QKV_head , QKV_seqlen
      );


    case 128:
      launch_flash_attn<128 , 2>(
        reinterpret_cast<half*>(Q.data_ptr()),
        reinterpret_cast<half* >(K.data_ptr()),
        reinterpret_cast<half* >(V.data_ptr()),
        reinterpret_cast<half* >(O.data_ptr()),
        QKV_batch ,QKV_head , QKV_seqlen
      );

    case 256:
      launch_flash_attn<256 ,2 >(
        reinterpret_cast<half*>(Q.data_ptr()),
        reinterpret_cast<half* >(K.data_ptr()),
        reinterpret_cast<half* >(V.data_ptr()),
        reinterpret_cast<half* >(O.data_ptr()),
        QKV_batch ,QKV_head , QKV_seqlen
      );

    default:
      throw std::runtime_error("Head dim is not fucking supported bitch!");
      break;
  }


  return O;
  
}

}