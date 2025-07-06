import math
import random


import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def data_from_df(df, target_level=None, label_pipeline=None):
    barcodes = df["nucleotides"].to_list()

    if target_level:
        species = df[target_level].to_list()
        species = np.array(list(map(label_pipeline, species)))

    print(f"[INFO]: There are {len(barcodes)} barcodes")
    # Number of training samples and entire data
    N = len(barcodes)

    # Reading barcodes and labels into python list
    labels = []

    for i in range(N):
        if len(barcodes[i]) > 0:
            barcodes.append(barcodes[i])
            if target_level:
                labels.append(species[i])

    sl = 660  # Max_length

    nucleotide_dict = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

    X = np.zeros((N, sl, 5), dtype=np.float32)  # Can't do zeros because
    for i in range(N):
        j = 0
        while j < min(sl, len(barcodes[i])):
            k = nucleotide_dict[barcodes[i][j]]
            X[i][j][k] = 1.0
            j += 1

    # print(X.shape, )
    return X, np.array(labels)


class MaskedOneHotDataset(Dataset):
    def __init__(self, X, mask_ratio=0.15, chunk_size=4):
        """
        X: numpy array of shape (N, L, 5), one-hot over {A,C,G,T,N}.
           Padding rows must be all zeros.
        mask_ratio: fraction of real tokens to mask
        chunk_size: length of each contiguous masked span
        """
        self.X = torch.from_numpy(X)  # (N, L, 5)
        self.mask_ratio = mask_ratio
        self.chunk_size = chunk_size

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        x = self.X[idx].clone()  # (L, 5)
        L = x.size(0)

        # 1) Padding mask: 1 for real tokens (sum>0), 0 for padding (zero-vector)
        att_mask = (x.sum(dim=-1) > 0).int()

        # 2) Targets: argmax over one-hot; set padding positions to -1
        targets = x.argmax(dim=-1)
        targets[att_mask == 0] = -1  # padding → -1

        # 3) Determine valid maskable positions: real A/C/G/T only
        valid = (targets >= 0) & (targets < 4)
        n_valid = valid.sum().item()
        n_chunks = math.ceil(self.mask_ratio * n_valid / self.chunk_size)

        # 4) Sample non-overlapping contiguous spans
        starts = []
        while len(starts) < n_chunks:
            s = random.randrange(0, L - self.chunk_size + 1)
            if valid[s] and all(abs(s - p) >= self.chunk_size for p in starts):
                starts.append(s)

        mask = torch.zeros(L, dtype=torch.bool)
        for s in starts:
            mask[s : s + self.chunk_size] = True
        mask &= valid  # ensure no padding or N masked

        # 5) Apply uniform fill (1/5) to masked positions
        x_masked = x.clone()
        x_masked[mask] = 1.0 / 5.0

        return x_masked, targets, att_mask, mask

import math
import random
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class HDF5MaskedDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        mask_ratio: float = 0.5,
        token_size: int = 4,
        randomize_offset: bool = True,
        pad_value: int = 255,
    ):
        """
        h5_path: HDF5 file with dataset "barcodes" shape (N, L), uint8
                 0–3=A/C/G/T, 4=real N, pad_value=padding
        mask_ratio: fraction of real {A,C,G,T} tokens to mask
        token_size: length of each contiguous masked span (and max offset)
        randomize_offset: whether to randomly shift sequence by up to token_size
        pad_value: integer sentinel in the HDF5 for padding
        """
        self.h5_path = h5_path
        self.mask_ratio = mask_ratio
        self.chunk_size = token_size
        self.randomize_offset = randomize_offset
        self.pad_value = pad_value
        self._file = None

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
            self.data = self._file["barcodes"]

    def __len__(self):
        with h5py.File(self.h5_path, "r") as f:
            return f["barcodes"].shape[0]

    def __getitem__(self, idx):
        # lazy-open HDF5
        self._ensure_open()

        # 1) Raw row and optional offset
        row = np.array(self.data[idx])  # shape (L,), uint8
        L = row.shape[0]
        if self.randomize_offset:
            offset = random.randint(0, self.chunk_size - 1)
        else:
            offset = 0
        if offset > 0:
            pad = np.full(offset, self.pad_value, dtype=row.dtype)
            row = np.concatenate([row[offset:], pad], axis=0)

        # 2) One-hot encode, then zero-out padding rows
        idxs = torch.from_numpy(row).long()
        clamped = idxs.clamp(max=4)
        x = F.one_hot(clamped, num_classes=5).float()  # (L,5)
        pad_mask = (idxs == self.pad_value)
        x[pad_mask] = 0.0

        # 3) Build att_mask & targets
        att_mask = (~pad_mask).int()  # 1 for real tokens, 0 for padding
        targets = idxs.clone()
        targets[pad_mask] = -1        # padding → -1

        # 4) Masking spans on A/C/G/T only
        valid = (targets >= 0) & (targets < 4)
        n_valid = int(valid.sum().item())
        n_chunks = math.ceil(self.mask_ratio * n_valid / self.chunk_size)

        starts = []
        while len(starts) < n_chunks:
            s = random.randrange(0, L - self.chunk_size + 1)
            if valid[s] and all(abs(s - p) >= self.chunk_size for p in starts):
                starts.append(s)

        mask = torch.zeros(L, dtype=torch.bool)
        for s in starts:
            mask[s : s + self.chunk_size] = True
        mask &= valid

        # 5) Apply uniform fill (1/5) to masked positions
        x_masked = x.clone()
        x_masked[mask] = 1.0 / 5.0

        return x_masked, targets, att_mask, mask

