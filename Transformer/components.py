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

        self.q = nn.Parameter(torch.randn((self.heads, embedding_dims, O_dim)))
        self.k = nn.Parameter(torch.randn((embedding_dims, O_dim))) #c1
        self.v = nn.Parameter(torch.randn((embedding_dims, O_dim))) #c2

        self.Wo = nn.Parameter(torch.randn((O_dim * self.heads, embedding_dims)))

    def forward(self, q, kv=None ,attn_mask=None):
        """
        q  shape : (B, S_q, E)
        kv shape : (B, S_k, E) or None

        If kv is None → self-attention: kv = q
        """

        if kv is None:
            kv = q

        B, S_q, E = q.shape
        _, S_k, _ = kv.shape

        q = q.unsqueeze(1)

        Q = q @ self.q
        K = kv @ self.k
        K = K.unsqueeze(1)
        V = kv @ self.v
        V = V.unsqueeze(1)
        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.O_dim)

        if self.mode == "masked":

            mask = torch.triu(torch.ones(S_q, S_k, device=scores.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))
            
        probs = torch.softmax(scores, dim=-1)
        context = probs @ V
        context = context.permute(0, 2, 1, 3).reshape(B, S_q, self.O_dim * self.heads)

        out = context @ self.Wo

        return out

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
    
class AddNorm(nn.Module):
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
        
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, unbiased=False, keepdim=True)

        norm = (x - mean) / torch.sqrt(var + self.eps)

        return norm * self.gamma + self.beta

