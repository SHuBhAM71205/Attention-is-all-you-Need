import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from Dataset.parallelDataSet import *
from Tokenizer import tokenizer
from Transformer import transformer,checkpoint
from Logger.logger import setup_logger

device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_dims = 512
d_ff = 2048
n_heads = 8
n_layers = 6
batch_size = 128
epochs = 5
label_smoothing = 0.1
step_counts = 0
warmup_steps = 4000

# test data path

en_bin="./Data/tokenized/dev_en.tok.bin"
en_idx="./Data/tokenized/dev_en.tok.idx"
hi_bin="./Data/tokenized/dev_hi.tok.bin"
hi_idx="./Data/tokenized/dev_hi.tok.idx"

mode = "local"

runtime_dir = "/model" if mode =="colab" else None
drive_dir = "./saves"

tknizer = tokenizer.Tokenizer(model_path=".",data_path="./Data/parallel-n/en-hi.all")

dataset = TokenizedParallelDataset(en_bin,en_idx,hi_bin,hi_idx)



en_hi = transformer.Transformer(
    tknizer,
    embedding_dims=embedding_dims,
    d_ff=d_ff,
    n_heads=n_heads,
    num_layers_enc=n_layers,
    num_layers_dec=n_layers,
    max_tokens=256,
    PATH="./saves"
).to(device)

pad_id = en_hi.pad_id

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, pad_id),
    num_workers=0,
)

total_params = sum(p.numel() for p in en_hi.parameters())

print(f"Total Model Parameters: {total_params:,}")
pnt = checkpoint.find_latest_checkpoint("./saves")


if pnt is None:
    print("There is no model to evaluate\n")
    exit(0)

chkpt = torch.load(pnt,map_location=device)

en_hi.load_state_dict(chkpt["model"])

en_hi.eval()
with torch.inference_mode():
    loss_batch = 0
    iterations = 0

    for en,hi in loader:
        en = en.to(device)
        hi = hi.to(device)
        
        logits,tgt_target = en_hi(en,hi)
        logits_flat = logits.reshape(-1, logits.size(-1))
        tgt_flat = tgt_target.reshape(-1)

        loss = F.cross_entropy(
            logits_flat,
            tgt_flat, 
            ignore_index=en_hi.pad_id
        )
        
        loss_batch += loss.item()
        iterations +=1
        
    print(f"The testing loss is {(loss_batch / iterations):.5f}")
