import torch
from torch import nn

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_models : int = 512,
                 num_heads : int = 8,
                 head_dim : int = 64):
        super().__init__()
        self.d_models = d_models
        self.num_heads = num_heads
        self.head_dim = head_dim

        assert d_models % num_heads == 0, f"Model dimensions should be divisible by number of heads ({d_models}, {num_heads})" 

        self.Projection = nn.Sequential(nn.Linear(d_models, d_models),
                                   nn.ReLU())
    
    def forward(self, query: torch.tensor, 
                key: torch.tensor,
                value: torch.tensor):
        batch_size = query.size(0)
        Q = self.Projection(query).view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, sequence_length, number of heads, head dimensions)
        K = self.Projection(key).view(batch_size, -1, self.num_heads, self.head_dim)
        V = self.Projection(value).view(batch_size, -1, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2) # (batch_size, number of heads, sequence_length, head dimensions)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scalar_dot = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        scalar_mask = torch.triu(torch.ones_like(scalar_dot), diagonal=1).bool()

        """
        [[False,  True,  True,  True],
          [False, False,  True,  True],
          [False, False, False,  True],
          [False, False, False, False]]
        
        """
        scalar_dot.masked_fill_(scalar_mask, float('-inf'))
        soft_max_output = torch.softmax(scalar_dot, dim=-1)
        # print(soft_max_output)

        attn_output = torch.matmul(soft_max_output, V)# (batch_size, number of heads, sequence_length, head dimensions)

        final_output = attn_output.transpose(0, 1).contiguous().view(batch_size, -1, self.d_models) 

        return final_output


if __name__ == "__main__":

    model = MaskedMultiHeadAttention(d_models= 4, num_heads=2, head_dim= 2)

    test_tensor = torch.randn(1, 4, 4)
    print("--------------------Encoder-----------------------")

    with torch.inference_mode():
        results = model(key = test_tensor, value = test_tensor, query = test_tensor)
        print("\n\nFinal Result:",results.shape)