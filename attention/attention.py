import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 512


class Transformers(nn.Module):
    pass



class MultiHeadAttention(nn.Module):


    def scaled_dot_product_attention(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor):
        d_k = 100  # ?????
        key_query = torch.matmul(key, query)
        scaled = key_query / torch.sqrt(d_k)


        if self.mask:
            pass 


        pass

    
    def forward(key, query, value):
        pass
    pass

