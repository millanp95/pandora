import math
import random

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
