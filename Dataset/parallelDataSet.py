import torch
from torch.utils.data import Dataset
import mmap
import array
import os

def load_offsets(path):
    arr = array.array("Q")
    itemsize = arr.itemsize  # 8 bytes for uint64

    filesize = os.path.getsize(path)
    n = filesize // itemsize

    with open(path, "rb") as f:
        arr.fromfile(f, n)

    return arr


def collate_fn(batch, tokenizer,device, max_len=None):
    en_texts = [x[0] for x in batch]
    hi_texts = [x[1] for x in batch]

    X = tokenizer.encode_batch(en_texts)
    Y = tokenizer.encode_batch(hi_texts)

    pad_id = tokenizer.sp.pad_id()

    if max_len is not None:
        X = [s[:max_len] for s in X]
        Y = [s[:max_len] for s in Y]

    max_x = max(len(s) for s in X)
    max_y = max(len(s) for s in Y)

    X = [s + [pad_id] * (max_x - len(s)) for s in X]
    Y = [s + [pad_id] * (max_y - len(s)) for s in Y]

    return (
        torch.tensor(X, dtype=torch.long),
        torch.tensor(Y, dtype=torch.long),
    )


class ParallelTextDataset(Dataset):

    def __init__(self, en_path, en_offset_path, hi_path, hi_offset_path):
        self.en_offsets = load_offsets(en_offset_path)
        self.hi_offsets = load_offsets(hi_offset_path)

        assert len(self.en_offsets) == len(self.hi_offsets)

        self.f_en = open(en_path, "rb")
        self.f_hi = open(hi_path, "rb")

        self.mm_en = mmap.mmap(self.f_en.fileno(), 0, access=mmap.ACCESS_READ)
        self.mm_hi = mmap.mmap(self.f_hi.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return len(self.en_offsets)

    def __getitem__(self, i):
        # EN
        start_en = self.en_offsets[i]
        end_en = (
            self.en_offsets[i + 1]
            if i + 1 < len(self.en_offsets)
            else self.mm_en.size()
        )
        en = self.mm_en[start_en:end_en].decode("utf-8").rstrip("\r\n")

        # HI
        start_hi = self.hi_offsets[i]
        end_hi = (
            self.hi_offsets[i + 1]
            if i + 1 < len(self.hi_offsets)
            else self.mm_hi.size()
        )
        hi = self.mm_hi[start_hi:end_hi].decode("utf-8").rstrip("\r\n")

        return en, hi

    def __del__(self):
        self.mm_en.close()
        self.mm_hi.close()
        self.f_en.close()
        self.f_hi.close()
