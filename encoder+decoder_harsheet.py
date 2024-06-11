import torch
import torch.nn as nn
import math
     
class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.embedding=nn.Embedding(d_model,vocab_size)
    def forward(self,x):
        return self.embedding(x)*math.sqrt(d_model)


class FeedForward(nn.Module):
    def __init__(self,d_model:int, d_ff:int,dropout:float):
        super().__init__()
        self.Linear_1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout(dropout=dropout)
        self.relu=nn.ReLU()
        self.Linear_2=nn.Linear(d_ff,d_model)
    def forward(self,x):
        return self.Linear_2(self.dropout(self.relu(self.Linear_1(x))))

class ResidualConnections(nn.Module):
    def __init__(self,dropout:float):
        super().__init__()
        self.dropout=nn.Dropout(dropout=dropout)
        self.norm=nn.LayerNorm()
    def forward(self,x):
        return x+self.dropout(self.norm(x))

class SelfAttentionBlock(nn.Module):
    def __init__(self,d_model:int,heads:int,dropout:float):
        super().__init__()
        self.d_k=d_model//heads
        assert d_model % heads == 0, f"Model dimensions should be divisible by number of heads"
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,q,k,v,mask):
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)
        # (Batch,Seq_Length,d_model)-->(Batch,Seq_Length,h,d_k)-->(Batch,h,Seq_Length,d_k
        query=query.view(query.shape[0],query.shape[1],self.heads,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.heads,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.heads,self.d_k).transpose(1,2)
        attn_score=torch.matmul(query,key.transpose(-2,-1))//math.sqrt(self.d_k)
        if mask is not None:
            attn_score.masked_fill(mask==0,-1e9)
        attn_score=attn_score.softmax(dim=-1)
        if dropout is not None:
            attn_score=dropout(attn_score)
        attn_product=torch.matmul(attn_score,value)
        # (Batch,h,Seq_Length,d_k)-->(Batch,Seq_Length,h,d_k)-->(Batch,Seq_Length,d_model)
        attn_product=attn_product.transpose(1,2).contiguous().view(attn_product.shape[0],-1,self.h*self.d_k)
        return self.w_o(attn_product)

class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block: SelfAttentionBlock,feed_forward_block:FeedForward,dropout:int):
        super().__init__()
        self.self_attn=self_attention_block
        self.ff=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnections(dropout) for _ in range(2)])
    def forward(self,x,src_mask):
        x=self.residual_connections[0](x,lambda x: self.self_attn(x,x,x,src_mask))
        x=self.residual_connections[1](x,self.ff)
        return x

class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layer_norm=nn.LayerNorm()
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x)
        return self.norm(x) 
        
class DecoderBlock(nn.Module):
    def __init__(self,self_attn_block:SelfAttentionBlock,cross_attn_block:SelfAttentionBlock,ff_block:FeedForward,dropout:float):
        super().__init__()
        self.residual_connections=nn.Module([ResidualConnections(dropout) for _ in range(3)])
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.residual_connections[0](x,lambda x:self.self_attn_block(x,x,x,tgt_mask))
        x=self.residual_connections[1](x,lambda x:self.self_attn_block(x,encoder_output,encoder_output,src_mask,tgt_mask))
        x=self.residual_connections[2](x,lambda x:self.ff_block)
        return x

class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.norm=nn.LayerNorm()
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

class FinalLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.final=nn.Linear(d_model,vocab_size)
    def forward(self,x):
        return torch.log_softmax(self.final(x),dim=-1)

class TransformerBlock(nn.Module):
    def __init__(self,)