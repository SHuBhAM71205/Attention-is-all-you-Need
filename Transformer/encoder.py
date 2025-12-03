from .components import * 

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dims, d_ff, n_heads):
        
        super().__init__()

        self.self_attn = Attention(
            embedding_dims=embedding_dims,
            O_dim=embedding_dims // n_heads,
            heads=n_heads,
            mode="multihead",
        )
        self.addnorm1 = AddNorm(embedding_dims)

        self.ffn = FFN(embedding_dims, d_ff)
        self.addnorm2 = AddNorm(embedding_dims)

    def forward(self, x, src_pad_mask=None):
        
        attn_mask = None
        
        if src_pad_mask is not None:
            attn_mask = src_pad_mask.unsqueeze(1).unsqueeze(2)
            
        att_out = self.self_attn(q=x, kv=x, attn_mask=attn_mask)
        x = self.addnorm1(x + att_out)

        
        ffn_out = self.ffn(x)
        x = self.addnorm2(x + ffn_out)

        return x
    
    

class Encoder(nn.Module):
    def __init__(self, embedding_dims, d_ff, n_heads, num_layers):
        
        super().__init__()

        self.layers = nn.ModuleList(
            EncoderLayer(embedding_dims, d_ff, n_heads)
            for _ in range(num_layers)
        )

    def forward(self, x, src_pad_mask=None):
        
        for layer in self.layers:
            x = layer(x, src_pad_mask=src_pad_mask)
        return x
