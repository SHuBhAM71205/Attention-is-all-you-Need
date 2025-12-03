
import torch 
import math
import torch.nn as nn

from .encoder import Encoder

from .decoder import Decoder 


class Transformer(nn.Module):
    def __init__(
        self,
        tokenizer,
        embedding_dims  =128,
        d_ff            =512,
        n_heads         =4,
        num_layers_enc  =2,
        num_layers_dec  =2,
        max_tokens      =512,
        PATH            = ".",
        load_from_saves =True,
        device = "cpu"
        
    ):
        '''
        :param PATH: It is used to pass the folder path where the model will be save/is saved
        '''
        super().__init__()
        
        self.PATH = PATH
        self.device = device
        try:
            self.load()
        except Exception as  e:
            pass
        
        self.tokenizer = tokenizer
        self.embedding_dims = embedding_dims
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.max_tokens = max_tokens

        self.vocab_size = tokenizer.sp.GetPieceSize()
        self.pad_id = tokenizer.sp.pad_id()
        self.unk_id = tokenizer.sp.unk_id()
        self.bos_id = tokenizer.sp.bos_id()
        self.eos_id = tokenizer.sp.eos_id()

        self.token_embedding = nn.Parameter(torch.randn(
            self.vocab_size, embedding_dims
        ) / math.sqrt(embedding_dims))

        self.pos_embedding = nn.Parameter(torch.randn(
            max_tokens, embedding_dims
        ) / math.sqrt(embedding_dims))

        self.encoder = Encoder(embedding_dims, d_ff, n_heads, num_layers_enc)
        self.decoder = Decoder(embedding_dims, d_ff, n_heads, num_layers_dec)

        self.W_out = nn.Parameter(torch.randn(size = (embedding_dims, self.vocab_size)) / math.sqrt(embedding_dims))
        self.b_out = nn.Parameter(torch.zeros(self.vocab_size))

    
    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, L) int64 tensor
        return:    (B, L, embedding_dims)
        """
        
        B, L = token_ids.shape

        if L > self.max_tokens:
            raise ValueError(f"Sequence length {L} exceeds max_tokens {self.max_tokens}")

        tok_emb = self.token_embedding[token_ids] 

        pos_ids = torch.arange(L, dtype=torch.long).unsqueeze(0).repeat(B, 1)
        pos_emb = self.pos_embedding[pos_ids] 

        return tok_emb + pos_emb
    
    def make_pad_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, L)
        return:    (B, L) boolean mask, True where PAD
        """
        return (token_ids == self.pad_id)
    
    def encode(self, src_ids: torch.Tensor):
        """
        src_ids: (B, S) int64 tensor of source token ids
        returns: enc_out (B, S, embedding_dims), src_pad_mask (B, S)
        """
        src_emb = self.embed(src_ids) 
        src_pad_mask = self.make_pad_mask(src_ids)

        enc_out = self.encoder(src_emb, src_pad_mask=src_pad_mask)
        return enc_out, src_pad_mask
    
    def decode(
        self,
        tgt_ids: torch.Tensor,
        enc_out: torch.Tensor,
        src_pad_mask: torch.Tensor,
    ):
        """
        tgt_ids:      (B, T)
        enc_out:      (B, S, embedding_dims)
        src_pad_mask: (B, S)  True on PAD
        returns: logits (B, T, vocab_size)
        """
        tgt_emb = self.embed(tgt_ids) 
        tgt_pad_mask = self.make_pad_mask(tgt_ids).to(self.device) 

        dec_out = self.decoder(
            tgt_emb,
            enc_out,
            src_pad_mask=src_pad_mask,
            tgt_pad_mask=tgt_pad_mask,
        )  

        logits = dec_out @ self.W_out + self.b_out

        return logits
    
    def infer(self, enc_out: torch.Tensor, src_pad_mask: torch.Tensor, max_new_tokens: int = 50):
        """
        Greedy decoding.

        enc_out:      (B, S, embedding_dims)
        src_pad_mask: (B, S)
        returns:      generated_ids (B, T_generated)
        """
        src_pad_mask.to(self.device)
        B = enc_out.shape[0]
        
        generated = torch.full((B, 1), self.bos_id, dtype=torch.long).to(self.device)

        for _ in range(max_new_tokens):
            logits = self.decode(generated, enc_out, src_pad_mask)

            next_logits = logits[:, -1, :]
            next_token  = torch.argmax(next_logits, dim=-1)

            next_token = next_token.unsqueeze(1).to(self.device)  # (B,1)
            generated  = torch.cat([generated, next_token], dim=1)

            if (next_token == self.eos_id).all():
                break

        return generated

    def forward(self, src_ids: torch.Tensor, tgt_ids= None):
        """
        src_ids: (B, S)
        tgt_ids: (B, T) or None

        If tgt_ids is provided -> training
        If tgt_ids is None     -> inference (greedy)
        """
        
        enc_out, src_pad_mask = self.encode(src_ids)
        
        
        if tgt_ids is None:
            return self.infer(enc_out, src_pad_mask)

        tgt_input  = tgt_ids[:, :-1]
        tgt_target = tgt_ids[:, 1:]

        logits = self.decode(tgt_input, enc_out, src_pad_mask)
        return logits, tgt_target
    
    def save(self):
        torch.save(self.state_dict(),f"{self.PATH}/{self.__class__.__name__}")
        
    def load(self):
        self.load_state_dict(torch.load(f"{self.PATH}/{self.__class__.__name__}"))