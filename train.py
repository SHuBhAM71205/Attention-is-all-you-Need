
from typing import Iterator
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader,IterableDataset
import Tokenizer.tokenizer as tk

from Transformer.transformer import Transformer

# device agnostic

device = "cuda" if torch.cuda.is_available() else "cpu"

## paths

dev_en_path = "./Data/dev_test/dev.en"
dev_hi_path = "./Data/dev_test/dev.hi"

train_en_path = "./Data/parallel-n/en-hi.en"
train_hi_path = "./Data/parallel-n/en-hi.hi"
# constants

embedding_dims = 512
d_ff           = 100
n_heads        = 5
n_layers       = 3
batch_size     = 200
epochs         = 100

## IterableDataSet

class ParallelTextData(IterableDataset):
    
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
                # yield raw text, no tokenization here
                yield en_line, hi_line
        

## reading train test data

with open(train_en_path,encoding='utf-8') as f:
    train_en = f.readlines()

with open(train_hi_path,encoding='utf-8') as f:
    train_hi = f.readlines()
    
train_en = [s.strip() for s in train_en]
train_hi = [s.strip() for s in train_hi]

## converting the sentence to tokens to token to the x y

tokenizer = tk.Tokenizer(".", "./Data/parallel-n/en-hi.all")

X = tokenizer.encode_batch(train_en)
Y = tokenizer.encode_batch(train_hi)

print(len(X))
## Tokenizer


# Transformer Model
en_hi = Transformer(
                        tokenizer,
                        embedding_dims= embedding_dims,
                        d_ff          = d_ff,
                        n_heads       = n_heads,
                        num_layers_enc= n_layers,
                        num_layers_dec= n_layers,
                        max_tokens    = 1000,
                        PATH          = "./saves",
                        device= device
                    ).to(device)



# Optimizer
optimizer = torch.optim.Adam(en_hi.parameters(),lr = 1e-3,)

# Training Loop
losses = []

for epoch in range(epochs):
    loss_batch = 0
    for batch_no in range(0,len(X),batch_size):
        
        batch_X_raw = X[batch_no:batch_no+batch_size]
        batch_Y_raw = Y[batch_no:batch_no+batch_size]
        
        max_len_X = len(max(batch_X_raw,key = len))
        max_len_Y = len(max(batch_Y_raw,key = len))
        
        batch_X = [seq + [0] * (max_len_X - len(seq)) for seq in batch_X_raw]
        batch_Y = [seq + [0] * (max_len_Y - len(seq)) for seq in batch_Y_raw]
        
        batch_X_tensor = torch.tensor(batch_X).to(device)
        batch_Y_tensor = torch.tensor(batch_Y).to(device)
        
        logits,tgt_target = en_hi(batch_X_tensor,batch_Y_tensor)
        
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