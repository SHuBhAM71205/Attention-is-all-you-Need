import torch
from torch.utils.data import Dataset
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


def collate_fn(batch, pad_id):
    xs, ys = zip(*batch)

    max_x = max(x.size(0) for x in xs)
    max_y = max(y.size(0) for y in ys)

    X = torch.full((len(xs), max_x), pad_id, dtype=torch.long)
    Y = torch.full((len(ys), max_y), pad_id, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, :x.size(0)] = x
        Y[i, :y.size(0)] = y

    return X, Y

class TokenizedParallelDataset(Dataset):
    def __init__(self, en_bin, en_idx, hi_bin, hi_idx):
        self.en_offsets = load_offsets(en_idx)
        self.hi_offsets = load_offsets(hi_idx)

        assert len(self.en_offsets) == len(self.hi_offsets)

        self.en_tokens = array.array("i")
        self.hi_tokens = array.array("i")

        with open(en_bin, "rb") as f:
            self.en_tokens.fromfile(f, os.path.getsize(en_bin) // 4)

        with open(hi_bin, "rb") as f:
            self.hi_tokens.fromfile(f, os.path.getsize(hi_bin) // 4)

    def __len__(self):
        return len(self.en_offsets)

    def __getitem__(self, i):
        en_start = self.en_offsets[i]
        en_end = self.en_offsets[i + 1] if i + 1 < len(self.en_offsets) else len(self.en_tokens)

        hi_start = self.hi_offsets[i]
        hi_end = self.hi_offsets[i + 1] if i + 1 < len(self.hi_offsets) else len(self.hi_tokens)

        return (
            torch.tensor(self.en_tokens[en_start:en_end], dtype=torch.long),
            torch.tensor(self.hi_tokens[hi_start:hi_end], dtype=torch.long),
        )
