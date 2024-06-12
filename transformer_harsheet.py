import torch
import torch.nn as nn
import math
     
class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.embedding=nn.Embedding(d_model,vocab_size)
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)


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
        if self.dropout is not None:
            attn_score=self.dropout(attn_score)
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
        x=self.residual_connections[1](x,self.ff(x))
        return x

class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layer_norm=nn.LayerNorm()
        self.layers=layers
    def forward(self,x,mask):
        self.mask=mask
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x) 
        
class DecoderBlock(nn.Module):
    def __init__(self,self_attn_block:SelfAttentionBlock,cross_attn_block:SelfAttentionBlock,ff_block:FeedForward,dropout:float):
        super().__init__()
        self.self_attn_block=self_attn_block
        self.cross_attn_block=cross_attn_block
        self.ff_block=ff_block
        self.residual_connections=nn.Module([ResidualConnections(dropout) for _ in range(3)])
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.residual_connections[0](x,lambda x:self.self_attn_block(x,x,x,tgt_mask))
        x=self.residual_connections[1](x,lambda x:self.cross_attn_block(x,encoder_output,encoder_output,src_mask,tgt_mask))
        x=self.residual_connections[2](x,lambda x:self.ff_block(x))
        return x

class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
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
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,finallayer:FinalLayer):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.finallayer=finallayer
    def encode(self,src,src_mask):
        src=self.src_embed(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)
    def decode(self,encoder_output,src_mask,tgt_mask):
        tgt=self.tgt_embed(tgt)
        tgt=self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    def final(self,x):
        return self.finallayer(x)
    
def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,d_model:int=512,N:int=6,h:int=8,dropout:float=0.1,d_ff:int=2048):
    src_embed=InputEmbeddings(d_model,src_vocab_size)
    tgt_embed=InputEmbeddings(d_model,tgt_vocab_size)
    src_pos=PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos=PositionalEncoding(d_model,tgt_seq_len,dropout)
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block=SelfAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForward(d_model,d_ff,dropout)
        encoder_block=EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)
        
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block=SelfAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block=SelfAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForward(d_model,d_ff,dropout)
        decoder_block=DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)
    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))
    final_layer=FinalLayer(d_model,tgt_vocab_size)
    transformer=TransformerBlock(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,final_layer)
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return transformer
    