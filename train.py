import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Logger.logger import setup_logger
# customs libs
import Tokenizer.tokenizer as tk

from Transformer.transformer import Transformer
from Transformer.checkpoint import save_checkpoint, find_latest_checkpoint
from Dataset.parallelDataSet import *

from DDP import ddp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.amp.grad_scaler import GradScaler


logs = True
logs_file_loc = "./logs"

logger = setup_logger("./logs")


# device agnostic
local_rank = ddp.setup_ddp()
device = torch.device(f"cuda:{local_rank}")
logger.info(f"Working on device {device}")

# paths 
# mode = str(input("Enter `colab` if working with the google colab \nEnter `local` if running locally \n"))
mode = "local"

en_bin="./Data/tokenized/en.tok.bin"
en_idx="./Data/tokenized/en.tok.idx"
hi_bin="./Data/tokenized/hi.tok.bin"
hi_idx="./Data/tokenized/hi.tok.idx"

runtime_dir = "/model" if mode =="colab" else None
drive_dir = "./saves"

# constants
embedding_dims = 512
d_ff = 2048
n_heads = 8
n_layers = 6
batch_size = 32
epochs = 5
label_smoothing = 0.1
step_counts = 0
warmup_steps = 4000
# tokenizer
tokenizer = tk.Tokenizer(".", "./Data/parallel-n/en-hi.all")
# Transformer Model

scaler = GradScaler(init_scale=2**10)

en_hi = Transformer(
        tokenizer,
        embedding_dims=embedding_dims,
        d_ff=d_ff,
        n_heads=n_heads,
        num_layers_enc=n_layers,
        num_layers_dec=n_layers,
        max_tokens=256,
        PATH="./saves"
    ).to(device)

en_hi = DDP(
    en_hi,
    device_ids= [local_rank],
    output_device=local_rank   
)

# DataLoader
dataset = TokenizedParallelDataset(en_bin,en_idx, hi_bin,hi_idx)

sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank = dist.get_rank(),
    shuffle=True
)


pad_id = en_hi.module.pad_id

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    collate_fn=lambda b: collate_fn(b, pad_id)
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
optimizer = torch.optim.Adam(en_hi.parameters(), lr=1.0,betas=(0.9,0.98),eps=1e-8) # dont try to make it 1.0 thats here cause of the custom lr scheduler

global_step = 0

if __name__ =="__main__":
    epoch_final = 0
    i = 0
    latest = find_latest_checkpoint(drive_dir)
    if latest:
        logger.info(f"Loading from checkpoint: {latest}   ")
        
        chkpt=torch.load(latest,map_location=device)
        en_hi.module.load_state_dict(chkpt["model"])
        optimizer.load_state_dict(chkpt["optimizer"])
        scaler.load_state_dict(chkpt["scaler"])
        i = chkpt["epoch"]
        global_step = chkpt["global_step"]
        
    else:
        
        logger.info("Starting fresh")
        global_step = 0

    save_every = 2000  # steps

    # Training Loop
    losses = []
    if dist.get_rank() == 0:
        logger.info(f"Started Trainnig Loop with epoch: {epochs} and batch size: {batch_size}\n")

    en_hi.train()
    for epoch in range(i,epochs):
        sampler.set_epoch(epoch)
        loss_batch = 0
        cnt = 0
        for src_ids, tgt_ids in loader:
            
            lr = lr_scheduler(embedding_dims, global_step, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
                
            src_ids = src_ids.to(device,non_blocking=True)
            tgt_ids = tgt_ids.to(device,non_blocking = True)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, tgt_target = en_hi(src_ids, tgt_ids)

                logits_flat = logits.reshape(-1, logits.size(-1))
                tgt_flat = tgt_target.reshape(-1)

            with torch.autocast(device_type="cuda",enabled=False):
                loss = F.cross_entropy(
                    logits_flat,
                    tgt_flat, 
                    ignore_index=en_hi.module.pad_id, 
                    label_smoothing=label_smoothing
                )
            
            loss_batch += loss.item()
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(en_hi.module.parameters(), 1.0)
            
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)

            scaler.update()
            
            global_step += 1        
            cnt+=1 

            if dist.get_rank()==0 and cnt % save_every == 0:
                
                save_checkpoint(
                        model = en_hi,
                        optimizer=optimizer,
                        scaler = scaler,
                        runtime_dir=runtime_dir,
                        drive_dir=drive_dir,
                        step=global_step,
                        epoch = epoch,
                        mode = mode
                    )
                if dist.get_rank() == 0:
                    logger.info(loss_batch / cnt)
            
        loss_batch /= cnt

        losses.append(loss_batch)
        if dist.get_rank() == 0:
            logger.info(f"Epoch {epoch} ; loss {loss_batch}")
        
        epoch_final = epoch
    
    save_checkpoint(
                        model = en_hi,
                        optimizer=optimizer,
                        runtime_dir=runtime_dir,
                        scaler = scaler,
                        drive_dir=drive_dir,
                        step=global_step,
                        epoch = epoch_final,
                        mode = mode
                    )
    
    
    ddp.cleanup_ddp()