import math
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embedding_dims, O_dim, heads=1, mode="multihead"):
        super().__init__()

        self.mode = mode
        self.O_dim = O_dim
        self.heads = heads

        self.flash_available = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )

        # ---- MQA: multi-query, single KV ----
        self.q_proj = nn.Linear(embedding_dims, heads * O_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dims, O_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dims, O_dim, bias=False)
        self.out_proj = nn.Linear(heads * O_dim, embedding_dims, bias=False)

    def forward(self, q, kv=None, attn_mask=None):
        if kv is None:
            kv = q

        B, S_q, _ = q.shape
        _, S_k, _ = kv.shape
        H, O = self.heads, self.O_dim

        Q = self.q_proj(q) \
                .view(B, S_q, H, O) \
                .permute(0, 2, 1, 3)          # (B,H,S_q,O)

        K = self.k_proj(kv) \
                .view(B, S_k, 1, O) \
                .permute(0, 2, 1, 3)          # (B,1,S_k,O)

        V = self.v_proj(kv) \
                .view(B, S_k, 1, O) \
                .permute(0, 2, 1, 3)          # (B,1,S_k,O)

        if self.flash_available:
            flash_mask = attn_mask

            if self.mode == "masked":
                causal = torch.tril(
                    torch.ones((1, 1, S_q, S_k), device=q.device)
                ).bool()
                flash_mask = flash_mask & causal if flash_mask is not None else causal

                invalid = ~flash_mask.any(dim=-1, keepdim=True)
                flash_mask[..., 0] |= invalid.squeeze(-1)

            out = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=flash_mask,
                dropout_p=0.1 if self.training else 0.0,
                is_causal=False,
                enable_gqa=True
            )
        else:
            raise RuntimeError("Non-flash path intentionally unsupported")

        # ---- CONCAT HEADS ----
        out = out.permute(0, 2, 1, 3).reshape(B, S_q, H * O)
        return self.out_proj(out)



class FFN(nn.Module):
    def __init__(self, embedding_dims, d_hidden):
        super().__init__()

        self.fc1 = nn.Linear(embedding_dims, d_hidden)
        self.fc2 = nn.Linear(d_hidden, embedding_dims)

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float16:
            with torch.autocast("cuda", enabled=False):
                x = x.float()
                hidden = torch.relu(self.fc1(x))
                out = self.fc2(hidden)
            return out.type_as(x)
        else:
            hidden = torch.relu(self.fc1(x))
            return self.fc2(hidden)

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
