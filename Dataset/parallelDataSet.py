import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from typing import Iterator


def collate_fn(batch, tokenizer, device, max_len=None):
    """
    batch: list of (en_str, hi_str)
    returns: src_ids (B,S), tgt_ids (B,T) tensors on CPU (move to GPU in loop)
    """
    en_texts = [x[0] for x in batch]
    hi_texts = [x[1] for x in batch]

    X = tokenizer.encode_batch(en_texts)
    Y = tokenizer.encode_batch(hi_texts)

    pad_id = tokenizer.sp.pad_id()

    if max_len is not None:
        X = [seq[:max_len] for seq in X]
        Y = [seq[:max_len] for seq in Y]

    max_len_X = max(len(s) for s in X)
    max_len_Y = max(len(s) for s in Y)

    X_pad = [s + [pad_id] * (max_len_X - len(s)) for s in X]
    Y_pad = [s + [pad_id] * (max_len_Y - len(s)) for s in Y]

    src_ids = torch.tensor(X_pad, dtype=torch.long)
    tgt_ids = torch.tensor(Y_pad, dtype=torch.long)

    return src_ids, tgt_ids

# IterableDataSet


class ParallelTextDataset(IterableDataset):

    def __init__(self, en_path, hi_path):
        super().__init__()

        self.en_path = en_path
        self.hi_path = hi_path

    def __iter__(self) -> Iterator:

        with open(self.en_path, encoding="utf-8") as f_en, \
                open(self.hi_path, encoding="utf-8") as f_hi:

            for en_line, hi_line in zip(f_en, f_hi):
                en_line = en_line.strip()
                hi_line = hi_line.strip()

                yield en_line, hi_line

