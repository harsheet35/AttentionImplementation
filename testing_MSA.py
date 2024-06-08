import torch

# Example dimensions and Q, K tensors
batch_size = 1
num_heads = 2
seq_length = 4
head_dim = 4

# Example Q and K tensors
Q = torch.randn(batch_size, num_heads, seq_length, head_dim)
K = torch.randn(batch_size, num_heads, seq_length, head_dim)

# Compute scalar_dot
scalar_dot = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))

# Create a mask for the upper triangle (excluding the diagonal)
mask = torch.triu(torch.ones_like(scalar_dot), diagonal=0).bool()

print(mask)

# Set the upper triangle to -inf
scalar_dot.masked_fill_(mask, float('-inf'))

# print(scalar_dot)