import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_MLM(nn.Module):
    def __init__(
        self,
        max_len: int,
        d_model: int = 768,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len

        # --- Single Conv block + Norm + Dropout + Pool ---
        self.conv = nn.Conv1d(5, d_model, kernel_size=4, stride=2, padding=1)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Compute downsampled length
        # After conv: ceil(max_len/2), after pool: ceil(prev/2)
        L1 = math.ceil(max_len / 2)
        L2 = math.ceil(L1 / 2)
        self.down_len = L2

        # --- Learned positional embeddings ---
        self.pos_emb = nn.Embedding(self.down_len, d_model)

        # --- BERT-style Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead - 1,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Upsampling via ConvTranspose1d x2 ---
        self.up1 = nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2)
        self.act_up1 = nn.GELU()
        self.up2 = nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2)
        self.act_up2 = nn.GELU()

        # --- Final classification head over 4 bases ---
        self.classifier = nn.Linear(d_model, 4)

    def forward(self, x: torch.Tensor, att_mask: torch.Tensor):
        """
        x: (B, L, 5) one-hot input
        att_mask: (B, L) attention mask (1 = valid, 0 = pad)
        """
        B, L, _ = x.shape

        # --- Conv stage ---
        h = x.transpose(1, 2)  # (B, 5, L)
        h = self.conv(h)  # (B, d_model, L1)
        h = h.transpose(1, 2)  # (B, L1, d_model)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = h.transpose(1, 2)  # (B, d_model, L1)
        h = self.pool(h)  # (B, d_model, down_len)

        # --- Add positional embeddings ---
        h = h.transpose(1, 2)  # (B, down_len, d_model)
        pos = self.pos_emb.weight.unsqueeze(0)  # (1, down_len, d_model)
        h = h + pos

        # --- Transformer ---
        # Downsample attention mask to match down_len
        factor = L // self.down_len
        att_ds = att_mask[:, ::factor] == 0
        z = self.transformer(
            h.permute(1, 0, 2), src_key_padding_mask=att_ds
        )  # (down_len, B, d_model)
        z = z.permute(1, 0, 2)  # (B, down_len, d_model)

        # --- Upsampling back to original resolution ---
        u = z.transpose(1, 2)  # (B, d_model, down_len)
        u = self.act_up1(self.up1(u))  # (B, d_model, down_len*2)
        u = self.act_up2(self.up2(u))  # (B, d_model, down_len*4)
        u = u.transpose(1, 2)  # (B, L_out, d_model)

        # --- Classification across 4 bases ---
        logits = self.classifier(u)  # (B, L_out, 4)
        return logits
