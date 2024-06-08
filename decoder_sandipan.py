import torch
from torch import nn
from encoder_sandipan import MultiHeadScalarAttention
from MaskedMultiHeadAttention import MaskedMultiHeadAttention

class Transformer_Decoder(nn.Module):
    def __init__(self, d_models: int = 512, num_heads: int = 8, head_dim: int = 64):
        super().__init__()

        self.d_models = d_models
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.MSA = MultiHeadScalarAttention(head_dim=head_dim, num_heads=num_heads, d_models=d_models)
        self.Masked_MSA = MaskedMultiHeadAttention(head_dim=head_dim, num_heads=num_heads, d_models=d_models)

        self.FeedForward = nn.Sequential(nn.Linear(d_models, d_models),
                                          nn.ReLU(),
                                          nn.Dropout(p = 0.1),
                                          nn.Linear(d_models, d_models),
                                          nn.ReLU())
        
        self.norm = nn.LayerNorm(d_models, d_models)

    def forward(self, x : torch.tensor,
                encoder_output: torch.tensor):
        
        mask_attention = self.Masked_MSA(query = x, key = x, value = x)
        x = mask_attention + x
        x = self.norm(x)

        multi_head_attention = self.MSA(query = x, key = encoder_output, value = encoder_output)
        x = multi_head_attention + x
        x = self.norm(x)

        FFN = self.FeedForward(x)
        x = x + FFN
        x = self.norm(x)

        return x



if __name__ == "__main__":
    transformer = Transformer_Decoder()
    tensor = torch.randn((1, 10, 512))
    print("The tensor shape is:", tensor.shape)
    print("--------------------Decoder-----------------------")

    with torch.inference_mode():
        results = transformer(x = tensor, encoder_output = tensor)
        print("\n\n\n Final Result:",results.shape)