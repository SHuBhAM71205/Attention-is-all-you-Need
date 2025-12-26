
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

import Tokenizer.tokenizer as tk

from Transformer.transformer import Transformer
from Transformer.checkpoint import find_latest_checkpoint
# device agnostic

device = "cuda" if torch.cuda.is_available() else "cpu"


test_en_path = "./Data/dev_test/test.en"
test_hi_path = "./Data/dev_test/test.hi"

embedding_dims = 128
d_ff = 100
n_heads = 4
n_layers = 2
batch_size = 128
epochs = 5
label_smoothing = 0.1

tokenizer = tk.Tokenizer(".", "./Data/parallel-n/en-hi.all")


with open(test_en_path,encoding='utf-8') as f:
    test_en = f.readlines()

with open(test_hi_path,encoding='utf-8') as f:
    test_hi = f.readlines()
    
    
tokenizer = tk.Tokenizer(".", "./Data/dev_test/test.all")

X = []
Y = []

for sentences in zip(test_en,test_hi):
    en_sentence = sentences[0]
    hi_sentence = sentences[1]
    X.append([4] + tokenizer.encode(en_sentence,encode_type='tokens') + [5]) # type: ignore
    Y.append([4] + tokenizer.encode(hi_sentence,encode_type='tokens') + [5]) # type: ignore
    # print(f"EN: {en_sentence} \nTOKENS: {en_token}\nHI: {hi_sentence} \nTOKENS: {hi_token}\n")


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

chkpt = find_latest_checkpoint("./saves")


latest_ckpt = torch.load(chkpt,map_location=device)

en_hi.load_state_dict(latest_ckpt["model"])
with torch.inference_mode():
    stmt = str(input("Enter the english statment:\n"))
    
    tkns = torch.tensor(tokenizer.encode(stmt,encode_type="tokens")).unsqueeze(0).to(device)
    
    gen_tkns = en_hi(tkns)
    
    print(f"Hindi: \n {tokenizer.decode(gen_tkns.tolist())}")
    
    pass