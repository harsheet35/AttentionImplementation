import torch
from torch import nn
from encoder_sandipan import TransformerEncoder
from decoder_sandipan import Transformer_Decoder

class Transformer_model(nn.Module):
    def __init__(self, head_dims: int = 64,
                 num_heads: int = 8,
                 d_models: int = 512,
                 vocab_size: int = 10000):
        super().__init__()

        self.encoder = TransformerEncoder(head_dim= head_dims, num_heads=num_heads, d_models=d_models)
        self.decoder = Transformer_Decoder(head_dim= head_dims, num_heads=num_heads, d_models=d_models)

        self.Linear = nn.Sequential(nn.Linear(d_models, vocab_size),
                               nn.ReLU())
        
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, inputs, outputs):
        encoder_outputs = self.encoder(inputs)
        decoder_output = self.decoder(x = outputs, encoder_output = encoder_outputs)

        x = self.Linear(decoder_output)
        x = self.softmax(x)

        return x


if __name__ ==  "__main__":
    model = Transformer_model(vocab_size=5)

    tensor = torch.randn(1, 10, 512)

    with torch.inference_mode():
        results = model(inputs = tensor, outputs = tensor)
        print(results)
        print(results.shape)