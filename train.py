import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Logger.logger import setup_logger
# customs libs
import Tokenizer.tokenizer as tk
from Transformer.transformer import Transformer
from Transformer.checkpoint import save_checkpoint, find_latest_checkpoint
from Dataset.parallelDataSet import *

logs = True
logs_file_loc = "./logs"

logger = setup_logger("./logs")

# device agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Working on device {device}")

# paths
# mode = str(input("Enter `colab` if working with the google colab \nEnter `local` if running locally \n"))
mode = "local"
train_en_path = "./Data/parallel-n/en-hi.en"
train_hi_path = "./Data/parallel-n/en-hi.hi"
train_en_offset_path = "./ByteOffsetGenerator/en_offset.bo"
train_hi_offset_path = "./ByteOffsetGenerator/hi_offset.bo"

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
step_counts = 0
warmup_steps = 4000
# tokenizer
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
dataset = ParallelTextDataset(train_en_path,train_en_offset_path, train_hi_path,train_hi_offset_path)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer, device, max_len=254),
    num_workers=0,
)

# Optimizer
## lr scheduler according to the Attention is all you need
def lr_scheduler(d_model,global_step,warmup_steps):
    global_step = max(global_step,1)
    lr = (d_model ** - 0.5) * min(
                                    global_step ** -0.5 ,
                                    global_step * (warmup_steps ** -1.5)
                                    )
    return lr
## optimizer
optimizer = torch.optim.Adam(en_hi.parameters(), lr=1.0,betas=(0.9,0.98),eps=1e-9) # dont try to make it 1.0 thats here cause of the custom lr scheduler

global_step = 0

if __name__ =="__main__":
    
    latest = find_latest_checkpoint(drive_dir)
    if latest:
        logger.info(f"Loading from checkpoint: {latest}   ")
        
        chkpt=torch.load(latest,map_location=device)
        en_hi.load_state_dict(chkpt["model"])
        optimizer.load_state_dict(chkpt["optimizer"])
        global_step = chkpt["global_step"]
        
    else:
        logger.info("Starting fresh")
        global_step = 0

    save_every = 2000  # steps

    # Training Loop
    losses = []
    logger.info(f"Started Trainnig Loop with epoch: {epochs} and batch size: {batch_size}\n")

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
            
            global_step += 1
            
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(
                                                    d_model=embedding_dims,
                                                    global_step=global_step , 
                                                    warmup_steps= warmup_steps
                                                    )
            optimizer.zero_grad()

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(en_hi.parameters(), 1.0)
            
            optimizer.step()

            cnt+=1 

            if global_step % save_every == 0:
                
                save_checkpoint(
                        model = en_hi,
                        optimizer=optimizer,
                        runtime_dir=runtime_dir,
                        drive_dir=drive_dir,
                        step=global_step,
                        mode = mode
                    )
                logger.info(loss_batch / cnt)
            
        loss_batch /= cnt

        losses.append(loss_batch)

        logger.info(f"Epoch {epoch} ; loss {loss_batch}")
        
        
    
    save_checkpoint(
                        model = en_hi,
                        optimizer=optimizer,
                        runtime_dir=runtime_dir,
                        drive_dir=drive_dir,
                        step=global_step,
                        mode = mode
                    )