import math
import torch
import torch.nn as nn

class Attention(nn.Module):

    def __init__(self, embedding_dims, O_dim, heads=1, mode="multihead"):
        """
        mode:
            - 'multihead'  (normal self / cross)
            - 'masked'     (causal self-attention)
        """
        
        super().__init__()
        
        self.mode  = mode
        self.O_dim = O_dim

        if mode == "multihead":
            self.heads = heads
        elif mode == "masked":
            self.heads = 1
        else:
            raise ValueError("Invalid mode: should be 'multihead' or 'masked'")

        self.flash_available = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        if self.flash_available :
            print("fast as flash attn ...")
            
        self.q = nn.Parameter(torch.empty((self.heads, embedding_dims, O_dim)))
        self.k = nn.Parameter(torch.empty((embedding_dims, O_dim))) 
        self.v = nn.Parameter(torch.empty((embedding_dims, O_dim))) 
        self.Wo = nn.Parameter(torch.empty((O_dim * self.heads, embedding_dims)))

        nn.init.xavier_uniform_(self.q)
        nn.init.xavier_uniform_(self.k)
        nn.init.xavier_uniform_(self.v)
        nn.init.xavier_uniform_(self.Wo)
        
    def forward(self, q, kv=None ,attn_mask=None):
        """
        q  shape : (B, S_q, E) contain the current and the previous tokens
        kv shape : (B, S_k, E) or None

        If kv is None → self-attention: kv = q
        """

        if kv is None:
            kv = q

        B, S_q, E = q.shape
        _, S_k, _ = kv.shape


        q = q.unsqueeze(1) #(B,1,S_q,E)
        
        Q = q @ self.q #(B,n_head,S_q or 1,O_dim) # multiple queries is generated
        
        K = kv @ self.k #(B,S_q,O_dim)
        
        K = K.unsqueeze(1) #(B,1,S_q,O_dim)
        
        V = kv @ self.v #(B,S_q,O_dim)
        
        V = V.unsqueeze(1) #(B,1,S_q,O_dim)
        
        mask = attn_mask
        if not self.flash_available and self.mode == "masked":
            causal_mask = torch.triu(torch.ones((S_q, S_k), device=q.device), diagonal=1).bool()
            mask = causal_mask if mask is None else (mask | causal_mask)
        
        if self.flash_available:
            
            out = torch.nn.functional.scaled_dot_product_attention(
                                                                    Q,K,V,
                                                                    attn_mask=attn_mask,
                                                                    dropout_p=0.2 if self.training else 0.0,
                                                                    is_causal= True if self.mode == "masked" else False,
                                                                    enable_gqa=True
                                                                )
        else:
            
            with torch.autocast(device_type="cuda",enabled=False):
                Q_f ,K_f,V_f= Q.float(),K.float(),V.float()
                
                scores = Q_f @ K_f.transpose(-1, -2) / math.sqrt(self.O_dim) #(B,n_head,S_q,S_q)

                scores = scores - scores.max(dim = -1, keepdims = True).values

                scores = scores.masked_fill(mask, float('-inf'))
                                
                probs = torch.softmax(scores, dim=-1) #(B,n_head,S_q,S_q)
                
                out = probs @ V_f #(B,n_head,S_q,o_dims)
        
        out = out.permute(0, 2, 1, 3).reshape(B, S_q, self.O_dim * self.heads)

        return (out.type_as(q) if not self.flash_available else out) @ self.Wo

class FFN(nn.Module):
    def __init__(self, embedding_dims, d_hidden):
        """
        embedding_dims  : embedding dimension (input/output)
        d_hidden : hidden layer dimension (usually 2–4x embedding_dims)
        """
        
        super().__init__()
        
        self.embedding_dims  = embedding_dims
        self.d_hidden = d_hidden

        self.W1 = nn.Parameter(torch.randn(size = (embedding_dims, d_hidden)) / math.sqrt(embedding_dims))
        self.b1 = nn.Parameter(torch.zeros(d_hidden))
        self.W2 = nn.Parameter(torch.randn(size = (d_hidden, embedding_dims)) / math.sqrt(d_hidden))
        self.b2 = nn.Parameter(torch.zeros(embedding_dims))


    def forward(self, x):
        """
        x: (B, S, embedding_dims)
        """
        hidden = torch.relu(x @ self.W1 + self.b1)

        out = hidden @ self.W2 + self.b2

        return out
    
class AddNorm(nn.Module): # this is typo where ever addNorm is found its just norm per layer i.e. LayerNorm 
    def __init__(self, embd_dim, eps=1e-6):
        
        super().__init__()
        
        self.gamma = nn.Parameter(torch.ones(size = (1, 1, embd_dim)))

        self.beta = nn.Parameter(torch.zeros(size = (1,1,embd_dim)))

        self.eps = eps
        
    def forward(self, x):
        
        '''
            espected x = x + y as both have same dims it is possible 

            and then applying the Batch norm layer

        '''
        with torch.autocast(device_type="cuda",enabled=False):
            x_fp32=x.float()
            mean = x_fp32.mean(dim=-1, keepdim=True)
            var  = x_fp32.var(dim=-1, unbiased=False, keepdim=True)

            norm = (x_fp32 - mean) / torch.sqrt(var + self.eps)

            out = norm * self.gamma.float() + self.beta.float()

        return out.type_as(x)
