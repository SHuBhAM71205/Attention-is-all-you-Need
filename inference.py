
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

import Tokenizer.tokenizer as tk

from Transformer.transformer import Transformer

# device agnostic

device = "cuda" if torch.cuda.is_available() else "cpu"


test_en_path = "./Data/dev_test/test.en"
test_hi_path = "./Data/dev_test/test.hi"

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


en_hi : Transformer = Transformer(
                        tokenizer,
                        embedding_dims  = 30,
                        d_ff            = 10,
                        n_heads         = 2,
                        num_layers_enc  = 1,
                        num_layers_dec  = 1,
                        max_tokens      = 1000,
                        PATH            = "./saves",
                        load_from_saves = True,
                        device= device
                    ).to(device)


with torch.inference_mode():
    stmt = str(input("Enter the english statment:\n"))
    
    tkns = torch.tensor(tokenizer.encode(stmt,encode_type="tokens")).unsqueeze(0).to(device)
    
    gen_tkns = en_hi(tkns)
    
    print(f"Hindi: \n {tokenizer.decode(gen_tkns.tolist())}")
    
    pass