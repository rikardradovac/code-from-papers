import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

d_model = 512


# all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512.

class Transformers(nn.Module):
    pass


class Encoder(nn.Module):
    def __init__(self, d_model: int, num_encoder_layers: int, num_heads: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        # stack N encoder blocks
        self.encoder_layers = nn.Sequential(
            *[EncoderLayer(d_model=d_model, num_heads=self.num_heads) for _ in range(self.num_encoder_layers)])

    def forward(self, x):
        return self.encoder_layers(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int = 2048) -> None:
        super().__init__()
        self.d_model = d_model

        self.layer_normalization = nn.LayerNorm((1, 30, 512))
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads)
        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model, ff_dim=ff_dim)

    def forward(self, x):
        residual = x
        x = self.multi_head_attention(x)
        x = self.layer_normalization(residual + x)
        residual = x
        x = self.feed_forward(residual + x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.linear = nn.Linear(d_model, d_model)
        self.head_dim = d_model // num_heads
        # input matrix turns into query, key and value of the same size
        self.query_key_value = nn.Linear(d_model, d_model * 3)

    @staticmethod
    def scaled_dot_product_attention(keys: torch.Tensor, query: torch.Tensor, values: torch.Tensor, mask: bool = False):
        d_k = keys.size()[-1]
        query = query.transpose(-1, 2)  # transpose for matmul
        print(keys.size(), query.size())
        key_query = torch.matmul(keys, query)
        key_query = key_query / np.sqrt(d_k)  # scaled

        if mask:
            key_query += mask

        key_query = F.softmax(key_query, dim=-1)

        result = torch.matmul(key_query, values)

        return result

    def forward(self, x):   # input is tensor of size (b_size, max_seq, d_model)
        batch_size, max_seq_length, d_model = x.size()

        query_key_value = self.query_key_value(x)
        # projected h times
        query_key_value = query_key_value.reshape(
            batch_size, self.num_heads, max_seq_length, self.head_dim * 3)  #(b_size, num_heads, max_seq, d_model)
        query, key, value = torch.tensor_split(query_key_value, 3, dim=-1)

        attention = self.scaled_dot_product_attention(
            query=query, keys=key, values=value)
        
        # concatenate the output from the dot attention, (batch_size, num_heads, max_seq, d_model * 3) --> (batch_size, max_seq_length, d_model * 3)
        concatenated = attention.view(
            batch_size, max_seq_length, self.num_heads * self.head_dim)
        result = self.linear(concatenated)

        return result


class PositionWiseFeedForward(nn.Module):
    def __init__(self, ff_dim, d_model):
        super().__init__()
        self.linear_in = nn.Linear(d_model, ff_dim)
        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(ff_dim, d_model)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.linear_out(x)
        return x


test_input = torch.rand((1, 30, 512))  # batch, seq, d_mod

# attention = MultiHeadAttention(512, 8)

# attention(test_input)

# encoder_layer = EncoderLayer(512, 8)

# output = encoder_layer(test_input)
# print(output, output.size())


encoder = Encoder(512, 6, 8)

print(encoder(test_input))
