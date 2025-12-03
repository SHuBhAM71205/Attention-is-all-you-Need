from .components import *
import torch
import torch.nn as nn

class  DecoderLayer(nn.Module):
    
    def __init__(self, embedding_dims, d_ff, n_heads):
        
        super().__init__()
        
        self.self_attn = Attention(
            embedding_dims=embedding_dims,
            O_dim=embedding_dims // n_heads,
            heads=n_heads,
            mode="masked",
        )
        self.addnorm1 = AddNorm(embedding_dims)

        self.cross_attn = Attention(
            embedding_dims=embedding_dims,
            O_dim=embedding_dims // n_heads,
            heads=n_heads,
            mode="multihead",
        )
        self.addnorm2 = AddNorm(embedding_dims)

        self.ffn = FFN(embedding_dims, d_ff)
        
        self.addnorm3 = AddNorm(embedding_dims)

    def forward(self, x, enc_out, src_pad_mask=None, tgt_pad_mask=None):
        """
        x      : (B, T, embedding_dims)  - decoder input states
        enc_out: (B, S, embedding_dims)  - encoder output states
        """ 

        self_attn_mask = None
        if tgt_pad_mask is not None:
            self_attn_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)

        self_attn_out = self.self_attn(q=x, kv=x, attn_mask=self_attn_mask)
        x = self.addnorm1(x + self_attn_out)

        cross_attn_mask = None
        if src_pad_mask is not None:
            cross_attn_mask = src_pad_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)

        cross_attn_out = self.cross_attn(q=x, kv=enc_out, attn_mask=cross_attn_mask)
        x = self.addnorm2(x + cross_attn_out)

        ffn_out = self.ffn(x)
        x = self.addnorm3(x + ffn_out)

        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dims, d_ff, n_heads, num_layers):
        
        super().__init__()
        
        self.layers = nn.ModuleList(
            DecoderLayer(embedding_dims, d_ff, n_heads)
            for _ in range(num_layers)
        )

    def forward(self, x, enc_out, src_pad_mask=None, tgt_pad_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
        return x
