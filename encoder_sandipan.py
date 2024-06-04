import torch
from torch import nn
print(torch.__version__)

MODEL_DIMENSION = 512
HEADS = 8
dk = MODEL_DIMENSION / HEADS
dv = MODEL_DIMENSION / HEADS

class MultiHeadScalarAttention(nn.Module):
    def __init__(self,
                 hidden_dim: int = 64,
                 num_heads: int = 64,
                 d_models: int = 512
    ):
        super().__init__()
        self.d_models = d_models

        assert hidden_dim % num_heads == 0, f"Model dimensions should be divisible by number of heads ({hidden_dim}, {num_heads})"
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, query, key, value):
        batch_size = query.size(0)

        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, num_heads, model_dimension)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, num_heads, model_dimension)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, num_heads, model_dimension)
        """
        The -1 in the view method is used as a placeholder to automatically infer the dimension size based on the remaining dimensions 
        and the total number of elements in the tensor.
        This is a feature of PyTorch's view method that allows for more flexible reshaping of tensors.
        """

        # Now we need to reshape the Q, K, V into (batch_size, num_heads, seq_length, model_dimension)
        Q = Q.transpose(1, 2) # (batch_size, num_heads, seq_length, head_dim)
        K = K.transpose(1, 2) # (batch_size, num_heads, seq_length, head_dim)
        V = V.transpose(1, 2) # (batch_size, num_heads, seq_length, head_dim)

        # Now carry on ...

        # Dot-Product
        """
        Note - for K.T we only want to change the last two dimensions
        (batch_size, seq_length, num_heads, model_dimension) ->  (batch_size, seq_length, model_dimension, num_heads)

        This is compatible for matrix multiplication because in Matrix multiplication we only multiply the last two dimensions and broad casting is done on the other dimensions
        """

        scalar_dot = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        softmax_weights = torch.softmax(scalar_dot, dim=-1)
        print(f"Softmax_weigths shape:{softmax_weights.shape}")

        atten_outputs = torch.matmul(softmax_weights, V)

        """
        We are dealing with a shape of (batch_size, num_heads, seq_length, head_dim)
        now we need to concatenate all heads 
        hence we need to reshape it to (num_heads, batch_size, seq_length, head_dim)

        and then reshape the it to (batch_size, seq_length, d_model) for further concatenation
        """

        concatenated_output = atten_outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.d_models)

        return concatenated_output

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dims: int, feed_forward_dims: int):
        super().__init__()

        self.FeedForward = nn.Sequential([nn.Linear(hidden_dims, feed_forward_dims),
                                          nn.Dropout(p = 0.1),
                                          nn.Linear(feed_forward_dims, hidden_dims)])
        self.MultiHeadAttention = MultiHeadScalarAttention()
        
        self.norm = nn.LayerNorm(hidden_dims, hidden_dims)
    
    def forward(self, x):
        MSA_Ouptut = self.MultiHeadAttention(query = x, key = x, value = x)
        x = torch.cat((MSA_Ouptut, x), dim=1)
        x = self.norm(x)
        FFN = self.FeedForward(x)
        encoder_output = torch.cat((FFN, x), dim=1)

        return encoder_output