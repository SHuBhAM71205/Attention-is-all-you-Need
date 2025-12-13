import torch
from torch.utils.data import DataLoader,IterableDataset
import torch.nn.functional as F

from Dataset.parallelDataSet import *
from Tokenizer import tokenizer
from Transformer import transformer,checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_dims = 128
d_ff = 100
n_heads = 4
n_layers = 2
batch_size = 64
epochs = 5
label_smoothing = 0.1

# test data path
dev_en_path = "./Data/dev_test/dev.en"
dev_hi_path = "./Data/dev_test/dev.hi"

runtime_dir = "/model"
drive_dir = "./saves"

tknizer = tokenizer.Tokenizer(model_path=".",data_path="./Data/parallel-n/en-hi.all")

dataset = ParallelTextDataset(dev_en_path,dev_hi_path)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=lambda batch: collate_fn(batch, tknizer, device, max_len=254),
    num_workers=0,
)

en_hi = transformer.Transformer(
    tknizer,
    embedding_dims=embedding_dims,
    d_ff=d_ff,
    n_heads=n_heads,
    num_layers_enc=n_layers,
    num_layers_dec=n_layers,
    max_tokens=256,
    PATH="./saves",
    device=device
).to(device)

pnt = checkpoint.find_latest_checkpoint("./saves")

if pnt is None:
    print("There is no model to evaluate\n")
    exit(0)

en_hi.load(pnt,map_location=device)


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
