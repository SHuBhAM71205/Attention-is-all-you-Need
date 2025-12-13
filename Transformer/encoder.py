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
        self.drop = Dropout(0.1)

    def forward(self, x, src_pad_mask=None):
        
        prob = torch.tensor([0.1,0.9])
        
        mode = torch.is_inference_mode_enabled()
        
        attn_mask = None
        
        if src_pad_mask is not None:
            attn_mask = src_pad_mask.unsqueeze(1).unsqueeze(2)
            
        x_norm = self.addnorm1(x)
        
        att_out = self.drop(self.self_attn(q=x_norm, kv=x_norm, attn_mask=attn_mask))
        
        x = x + att_out
        
        x_norm = self.addnorm2(x)
        ffn_out = self.drop(self.ffn(x_norm))

        x = x + ffn_out

        return x
    
    

class Encoder(nn.Module):
    def __init__(self, embedding_dims, d_ff, n_heads, num_layers):
        
        super().__init__()

        self.layers = nn.ModuleList(
            EncoderLayer(embedding_dims, d_ff, n_heads)
            for _ in range(num_layers)
        )
        
        self.final_norm = nn.LayerNorm(embedding_dims)

    def forward(self, x, src_pad_mask=None):
        
        for layer in self.layers:
            x = layer(x, src_pad_mask=src_pad_mask)
        return self.final_norm(x)


class Dropout(nn.Module):
    
    def __init__(self,dropout_rate=0.1):
        super().__init__()
        self.droupout_rate = dropout_rate
        
    def forward(self,x: torch.Tensor):
        mode = torch.is_inference_mode_enabled() or not self.training
        
        if mode is not True:
            
            mask = torch.bernoulli(torch.full(size = x.shape , fill_value= 1-self.droupout_rate , device = 'cuda' if torch.cuda.is_available() else 'cpu'))
            
            return x * mask / (1 - self.droupout_rate)
        
        return x