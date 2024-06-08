import torch
from torch import nn
print(torch.__version__)

MODEL_DIMENSION = 512
HEADS = 8
dk = MODEL_DIMENSION / HEADS
dv = MODEL_DIMENSION / HEADS

class MultiHeadScalarAttention(nn.Module):
    def __init__(self,
                 head_dim: int = 64,
                 num_heads: int = 8,
                 d_models: int = 512
    ):
        super().__init__()
        self.d_models = d_models
        self.num_heads = num_heads
        self.head_dim = head_dim

        assert d_models % num_heads == 0, f"Model dimensions should be divisible by number of heads ({d_models}, {num_heads})"
        self.query = nn.Sequential(nn.Linear(d_models, d_models),
                                          nn.ReLU())
        self.key = nn.Sequential(nn.Linear(d_models, d_models),
                                          nn.ReLU())
        self.value = nn.Sequential(nn.Linear(d_models, d_models),
                                          nn.ReLU())
    
    def forward(self, query, key, value):
        batch_size = query.size(0)

        Q = self.query(query)
        # print("Query 1: Shape", {Q.shape})
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, num_heads, model_dimension)
        # print("Query 1: Shape", {Q.shape})
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, num_heads, model_dimension)
        # print("Key 1: Shape", {K.shape})
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, num_heads, model_dimension)
        # print("value 1: Shape", {V.shape})
        """
        The -1 in the view method is used as a placeholder to automatically infer the dimension size based on the remaining dimensions 
        and the total number of elements in the tensor.
        This is a feature of PyTorch's view method that allows for more flexible reshaping of tensors.
        """
        # print("\n\n----------------------- Transposing the Query, Key, Values -----------------------------------\n")
        # Now we need to reshape the Q, K, V into (batch_size, num_heads, seq_length, model_dimension)
        Q = Q.transpose(1, 2) # (batch_size, num_heads, seq_length, head_dim)
        # print("Shape of Query after transposing:", Q.shape)
        K = K.transpose(1, 2) # (batch_size, num_heads, seq_length, head_dim)
        # print("Shape of Key after transposing:", K.shape)
        V = V.transpose(1, 2) # (batch_size, num_heads, seq_length, head_dim)
        # print("Shape of Valye after transposing:", V.shape)
        # Now carry on ...

        # Dot-Product
        """
        Note - for K.T we only want to change the last two dimensions
        (batch_size, seq_length, num_heads, model_dimension) ->  (batch_size, seq_length, model_dimension, num_heads)

        This is compatible for matrix multiplication because in Matrix multiplication we only multiply the last two dimensions and broad casting is done on the other dimensions
        """
        # print("\n\n---------------------------Taking dot product of the Query and Value-----------------------------\n")
        scalar_dot = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # print("Dimensions of the scalar product of key and query:", scalar_dot.shape)
        
        softmax_weights = torch.softmax(scalar_dot, dim=-1)
        # print(f"Softmax_weigths shape:{softmax_weights.shape}")

        atten_outputs = torch.matmul(softmax_weights, V)
        # print("Shape after multiplying the softmax result with the value:", (atten_outputs.shape))

        """
        We are dealing with a shape of (batch_size, num_heads, seq_length, head_dim)
        now we need to concatenate all heads 
        hence we need to reshape it to (num_heads, batch_size, seq_length, head_dim)

        and then reshape the it to (batch_size, seq_length, d_model) for further concatenation
        """
        # print("\n\n ---------------------- Concatenating all heads ---------------------------------\n")
        concatenated_output = atten_outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.d_models)
        # print("Shape after concatenating all outputs:", concatenated_output.shape)

        return concatenated_output

class TransformerEncoder(nn.Module):
    def __init__(self,
                 head_dim: int = 64,
                 num_heads: int = 8,
                 d_models: int = 512):
        super().__init__()

        self.FeedForward = nn.Sequential(nn.Linear(d_models, d_models),
                                          nn.ReLU(),
                                          nn.Dropout(p = 0.1),
                                          nn.Linear(d_models, d_models),
                                          nn.ReLU())
        self.MultiHeadAttention = MultiHeadScalarAttention(head_dim=head_dim, d_models=d_models, num_heads=num_heads)
        
        self.norm = nn.LayerNorm(d_models, d_models)
    
    def forward(self, x):
        MSA_Ouptut = self.MultiHeadAttention(query = x, key = x, value = x)
        # print("\n\n MSA_Output result shape:", MSA_Ouptut.shape)
        x = MSA_Ouptut + x
        # print("\n\n Shape after concatenating:", x.shape)
        x = self.norm(x)
        # print("\n\n Shape after normalisation:", x.shape)
        FFN = self.FeedForward(x)
        # print("FFN Output shape:", FFN.shape)
        encoder_output = FFN + x
        # print("Encoder output shape",encoder_output.shape)
        return encoder_output
    

if __name__ == "__main__":

    MSA = TransformerEncoder()

    tensor = torch.randn((1, 10, 512))
    print("The tensor shape is:", tensor.shape)

    with torch.inference_mode():
        results = MSA(tensor)
        print("\n\n\n Final Result:",results.shape)