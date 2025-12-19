
from typing import Iterator
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

# customs libs
import Tokenizer.tokenizer as tk
from Transformer.transformer import Transformer
from Transformer.checkpoint import save_checkpoint, find_latest_checkpoint
from Dataset.parallelDataSet import *

# device agnostic

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Working on device {device}")

# paths

mode = str(input("Enter `colab` if working with the google colab \nEnter `local` if running locally \n"))


train_en_path = "./Data/parallel-n/en-hi.en"
train_hi_path = "./Data/parallel-n/en-hi.hi"


runtime_dir = "/model" if mode =="colab" else None
drive_dir = "./saves"

# constants

embedding_dims = 128
d_ff = 100
n_heads = 4
n_layers = 2
batch_size = 128
epochs = 5
label_smoothing = 0.1

tokenizer = tk.Tokenizer(".", "./Data/parallel-n/en-hi.all")

# Transformer Model
en_hi = Transformer(
    tokenizer,
    embedding_dims=embedding_dims,
    d_ff=d_ff,
    n_heads=n_heads,
    num_layers_enc=n_layers,
    num_layers_dec=n_layers,
    max_tokens=256,
    PATH="./saves",
    device=device
).to(device)

# DataLoader
dataset = ParallelTextDataset(train_en_path, train_hi_path)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=lambda batch: collate_fn(batch, tokenizer, device, max_len=254),
    num_workers=0,
)

# Optimizer
optimizer = torch.optim.Adam(en_hi.parameters(), lr=1e-3,)

if __name__ =="__main__":
    

    latest = find_latest_checkpoint(drive_dir)
    if latest:
        print("Loading from checkpoint:", latest)
        en_hi.load(latest, map_location="cuda")

        global_step = int(latest.split("_step_")[1].split("_")[0])
    else:
        print("Starting fresh")
        global_step = 0

    save_every = 2000  # steps

    # Training Loop
    losses = []
    print(
        f"Started Trainnig Loop with epoch: {epochs} and batch size: {batch_size}\n")
    

    en_hi.train()
    for epoch in range(epochs):
        loss_batch = 0
        cnt = 0
        for src_ids, tgt_ids in loader:
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

            logits, tgt_target = en_hi(src_ids, tgt_ids)

            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_target.reshape(-1)

            loss = F.cross_entropy(
                logits_flat,
                tgt_flat, 
                ignore_index=en_hi.pad_id, 
                label_smoothing=label_smoothing
            )
            
            loss_batch += loss.item()
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            global_step += 1
            cnt+=1

            if global_step % save_every == 0:
                
                save_checkpoint(en_hi, runtime_dir, drive_dir, global_step,mode=mode)
                print(loss_batch / cnt)

        loss_batch /= batch_size

        losses.append(loss_batch)

        print(f"Epoch {epoch} ; loss {loss_batch}")
    
    
    save_checkpoint(en_hi, runtime_dir, drive_dir, global_step,mode = mode)


