import os
import mmap
import array
from tqdm import tqdm

# ---------------- CONFIG ----------------
EN_TXT = "./Data/parallel-n/en-hi.en"
HI_TXT = "./Data/parallel-n/en-hi.hi"

EN_IDX = "./ByteOffsetGenerator/en_offset.bo"
HI_IDX = "./ByteOffsetGenerator/hi_offset.bo"

OUT_DIR = "./Data/tokenized"
MAX_LEN = 256

os.makedirs(OUT_DIR, exist_ok=True)
# ---------------------------------------

# EN_TXT = "./Data/dev_test/en-hi.en"
# HI_TXT = "./Data/dev_test/en-hi.hi"

# EN_IDX = "./ByteOffsetGenerator/en_offset.bo"
# HI_IDX = "./ByteOffsetGenerator/hi_offset.bo"

# OUT_DIR = "./Data/tokenized"

def load_offsets(path):
    arr = array.array("Q")
    with open(path, "rb") as f:
        arr.fromfile(f, os.path.getsize(path) // arr.itemsize)
    return arr


def pretokenize(txt_path, idx_path, tokenizer, out_prefix):
    offsets = load_offsets(idx_path)

    f = open(txt_path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    tok_bin = open(out_prefix + ".tok.bin", "wb")
    tok_idx = open(out_prefix + ".tok.idx", "wb")

    token_buffer = array.array("i")   
    offset_buffer = array.array("Q")  

    cursor = 0

    for i in tqdm(range(len(offsets)), desc=f"Tokenizing {out_prefix}"):
        start = offsets[i]
        end = offsets[i + 1] if i + 1 < len(offsets) else mm.size()

        text = mm[start:end].decode("utf-8").rstrip("\r\n")

        ids = tokenizer.encode(text,"tokens")
        ids = ids[:MAX_LEN]

        offset_buffer.append(cursor)
        token_buffer.extend(ids)
        cursor += len(ids)

    token_buffer.tofile(tok_bin)
    offset_buffer.tofile(tok_idx)

    tok_bin.close()
    tok_idx.close()
    mm.close()
    f.close()


if __name__ == "__main__":
    import tokenizer as tk   

    tokenizer =tk.Tokenizer(".", "./Data/parallel-n/en-hi.all")

    pretokenize(
        EN_TXT,
        EN_IDX,
        tokenizer,
        os.path.join(OUT_DIR, "en")
    )

    pretokenize(
        HI_TXT,
        HI_IDX,
        tokenizer,
        os.path.join(OUT_DIR, "hi")
    )
