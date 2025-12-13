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
        
        self.drop = Dropout(0.1)

    def forward(self, x, enc_out, src_pad_mask=None, tgt_pad_mask=None):
        """
        x      : (B, T, embedding_dims)  - decoder input states
        enc_out: (B, S, embedding_dims)  - encoder output states
        """ 

        self_attn_mask = None
        if tgt_pad_mask is not None:
            self_attn_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)

        x_norm = self.addnorm1(x)
        self_attn_out = self.drop(self.self_attn(q=x_norm, kv=x_norm, attn_mask=self_attn_mask))
        x = x + self_attn_out

        cross_attn_mask = None
        if src_pad_mask is not None:
            cross_attn_mask = src_pad_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)

        
        x_norm = self.addnorm2(x)
        cross_attn_out = self.drop(self.cross_attn(q=x_norm, kv=enc_out, attn_mask=cross_attn_mask))
        x = x + cross_attn_out

        
        x_norm = self.addnorm3(x)
        ffn_out = self.drop(self.ffn(x_norm))
        x = x + ffn_out

        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dims, d_ff, n_heads, num_layers):
        
        super().__init__()
        
        self.layers = nn.ModuleList(
            DecoderLayer(embedding_dims, d_ff, n_heads)
            for _ in range(num_layers)
        )

        self.final_norm = nn.LayerNorm(embedding_dims)

    def forward(self, x, enc_out, src_pad_mask=None, tgt_pad_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
        return self.final_norm(x)


class Dropout(nn.Module):
    
    def __init__(self,dropout_rate=0.2):
        super().__init__()
        self.droupout_rate = dropout_rate
        
    def forward(self,x: torch.Tensor):
        mode = torch.is_inference_mode_enabled() or not self.training
        
        if mode is not True:
            
            mask = torch.bernoulli(torch.full(size = x.shape , fill_value= 1-self.droupout_rate , device = 'cuda' if torch.cuda.is_available() else 'cpu'))
            
            return x * mask / (1 - self.droupout_rate)
        
        return x