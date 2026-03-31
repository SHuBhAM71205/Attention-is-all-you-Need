import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from Dataset.parallelDataSet import *
from Tokenizer import tokenizer
from Transformer import transformer,checkpoint
from Logger.logger import setup_logger

from Score.BLEU import BLEU

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


def safe_decode(token_ids):
    if not token_ids:
        return ""
    return tknizer.decode(token_ids)



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


bleu_scoring = BLEU(4)

en_hi.load_state_dict(chkpt["model"])

en_hi.eval()
with torch.inference_mode():
    loss_batch = 0
    iterations = 0
    ppl_lst = []
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
        ppl = torch.exp(loss)
        ppl_lst.append(ppl.item())
        loss_batch += loss.item()
        iterations +=1
        
    print(f"The testing loss is {(loss_batch / iterations):.5f} ppl is {sum(ppl_lst)/len(ppl_lst):.5f}")
    
print("BLEU scoring>....")

en_hi.eval()
with torch.inference_mode():
    bleu_lst = []
    iterations = 0
    for en,hi in loader:
        en = en.to(device)
        hi = hi.to(device)
        generated_tokens,time= en_hi(en)
        
        generated_tokens = [
            [token for token in str_lst if token not in {tknizer.sp.pad_id(), tknizer.sp.bos_id(), tknizer.sp.eos_id()}] for str_lst in generated_tokens.tolist()
        ]
        
        hi_tokens = [
            [token for token in str_lst if token not in {tknizer.sp.pad_id(), tknizer.sp.bos_id(), tknizer.sp.eos_id()}] for str_lst in hi.tolist()
        ]
        
        gen_strs = [safe_decode(gen_tkns_i) for gen_tkns_i in generated_tokens]
        hi_strs = [safe_decode(hi_tkns) for hi_tkns in hi_tokens]
        
        bleu_lst.append(bleu_scoring.get_score(gen_strs,hi_strs))
    
print(f"BLEU score: {sum(bleu_lst)/len(bleu_lst):.5f}")
