
from typing import Iterator
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader,IterableDataset

## customs libs
import Tokenizer.tokenizer as tk
from Transformer.transformer import Transformer

# device agnostic

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Working on device {device}")

## paths

dev_en_path = "./Data/dev_test/dev.en"
dev_hi_path = "./Data/dev_test/dev.hi"

train_en_path = "./Data/parallel-n/en-hi.en"
train_hi_path = "./Data/parallel-n/en-hi.hi"

# constants

embedding_dims = 128
d_ff           = 100
n_heads        = 4
n_layers       = 2
batch_size     = 16
epochs         = 100

# functions

def collate_fn(batch, tokenizer, device, max_len=None):
    """
    batch: list of (en_str, hi_str)
    returns: src_ids (B,S), tgt_ids (B,T) tensors on CPU (move to GPU in loop)
    """
    en_texts = [x[0] for x in batch]
    hi_texts = [x[1] for x in batch]

    X = tokenizer.encode_batch(en_texts) 
    Y = tokenizer.encode_batch(hi_texts)

    pad_id = tokenizer.sp.pad_id()

    if max_len is not None:
        X = [seq[:max_len] for seq in X]
        Y = [seq[:max_len] for seq in Y]

    max_len_X = max(len(s) for s in X)
    max_len_Y = max(len(s) for s in Y)

    X_pad = [s + [pad_id] * (max_len_X - len(s)) for s in X]
    Y_pad = [s + [pad_id] * (max_len_Y - len(s)) for s in Y]

    src_ids = torch.tensor(X_pad, dtype=torch.long)
    tgt_ids = torch.tensor(Y_pad, dtype=torch.long)

    return src_ids, tgt_ids

## IterableDataSet

class ParallelTextDataset(IterableDataset):
    
    def __init__(self,en_path,hi_path):
        super().__init__()
        
        self.en_path = en_path
        self.hi_path = hi_path
        
    def __iter__(self) -> Iterator:
        
        with open(self.en_path, encoding="utf-8") as f_en, \
            open(self.hi_path, encoding="utf-8") as f_hi:
            
            for en_line, hi_line in zip(f_en, f_hi):
                en_line = en_line.strip()
                hi_line = hi_line.strip()
                
                yield en_line, hi_line
        

## converting the sentence to tokens to token to the x y

tokenizer = tk.Tokenizer(".", "./Data/parallel-n/en-hi.all")

# Transformer Model
en_hi = Transformer(
                        tokenizer,
                        embedding_dims= embedding_dims,
                        d_ff          = d_ff,
                        n_heads       = n_heads,
                        num_layers_enc= n_layers,
                        num_layers_dec= n_layers,
                        max_tokens    = 256,
                        PATH          = "./saves",
                        device= device
                    ).to(device)

# DataLoader

dataset = ParallelTextDataset(train_en_path, train_hi_path)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=lambda batch: collate_fn(batch, tokenizer, device,max_len=254),
    num_workers=0,
)

# Optimizer
optimizer = torch.optim.Adam(en_hi.parameters(),lr = 1e-3,)

# Training Loop
losses = []
print(f"Started Trainnig Loop with epoch: {epochs} and batch size: {batch_size}\n")
for epoch in range(epochs):
    loss_batch = 0
    
    for src_ids, tgt_ids in loader:
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)

        logits,tgt_target = en_hi(src_ids,tgt_ids)
        

        logits_flat = logits.reshape(-1, logits.size(-1))
        tgt_flat    = tgt_target.reshape(-1)

        loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=en_hi.pad_id)
        loss_batch+= loss.item()
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
    
    loss_batch /= batch_size
    
    losses.append(loss_batch)
    
    print(f"Epoch {epoch} ; loss {loss_batch}")

en_hi.save()