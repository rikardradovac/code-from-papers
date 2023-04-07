import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

d_model = 512


#all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512.

class Transformers(nn.Module):
    pass


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        
        self.layer_normalization = nn.LayerNorm((1, 30, 512))
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    
    def forward(self, x):
        residual = x
        x = self.multi_head_attention(x)
        print(x)

        x = self.layer_normalization(residual + x)
        print(x)




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.linear = nn.Linear(d_model, d_model)
        self.head_dim  = d_model // num_heads
        self.query_key_value = nn.Linear(d_model, d_model * 3) ## input matrix turns into query, key and value of the same size


    @staticmethod
    def scaled_dot_product_attention(keys: torch.Tensor, query: torch.Tensor, values: torch.Tensor, mask: bool = False):
        d_k = keys.size()[-1]
        query = query.transpose(-1, 2)  # transpose for matmul
        print(keys.size(), query.size())
        key_query = torch.matmul(keys, query)
        key_query = key_query / np.sqrt(d_k) ## scaled

        if mask:
            key_query += mask
        
        key_query = F.softmax(key_query, dim=-1)

        result = torch.matmul(key_query, values)

        return result

    
    def forward(self, x):   # input is tensor of size (b_size, max_seq, d_model)
        batch_size, max_seq_length, d_model = x.size()

        query_key_value = self.query_key_value(x)
        ## projected h times
        query_key_value = query_key_value.reshape(batch_size, self.num_heads, max_seq_length, self.head_dim * 3)
        query, key, value = torch.tensor_split(query_key_value, 3, dim=-1)
        
        print("query size", query.size())
        attention = self.scaled_dot_product_attention(query=query, keys=key, values=value)
        concatenated = attention.view(batch_size, max_seq_length, self.num_heads * self.head_dim)
        print("attention", attention.size())
        print("conc", concatenated.size())

        result = self.linear(concatenated)
        
        return result


test_input = torch.rand((1, 30, 512)) ##batch, seq, d_mod

# attention = MultiHeadAttention(512, 8)

# attention(test_input)

encoder_layer = EncoderLayer(512, 8)

encoder_layer(test_input)