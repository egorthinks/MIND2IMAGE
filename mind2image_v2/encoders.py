"""EEG Conformer-encoder dlya MIND2IMAGE 2.0.

Zamenyaet EEGLSTM_Encoder / EEGTransformer / EEG_CNN iz models/model.py.

Klyuchevye otlichiya ot 1.0:
  * prostranstvenno-vremennoy svyortochnyy tokenizator (kak v EEG Conformer);
  * pozicionnoe kodirovanie + CLS-token (v 1.0 bralsya posledniy token bez pos-enc);
  * subject-embedding dlya mezhsub"ektnoy obobshchaemosti;
  * vykhod razmernosti emb_dim == CLIP image embedding dim (768 dlya ViT-L/14).
"""

from __future__ import annotations

from typing import Optional
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Klassicheskoe sinusoidal'noe pozicionnoe kodirovanie."""

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class EEGConformer(nn.Module):
    """Conv-tokenizer + Transformer-encoder.

    Vkhod:  (B, C, T)  — kanaly x vremya.
    Vykhod: (B, emb_dim) — brain embedding (CLS).
    """

    def __init__(
        self,
        in_channels: int = 128,
        emb_dim: int = 768,
        depth: int = 6,
        heads: int = 8,
        temporal_kernel: int = 25,
        pool_kernel: int = 15,
        pool_stride: int = 5,
        n_filters: int = 40,
        dropout: float = 0.3,
        n_subjects: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim

        # prostranstvenno-vremennoy tokenizator: vkhod traktuetsya kak (B, 1, C, T)
        self.tokenizer = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, temporal_kernel),
                      padding=(0, temporal_kernel // 2)),       # vremennoy
            nn.Conv2d(n_filters, n_filters, (in_channels, 1)),  # prostranstvennyy (po kanalam)
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            nn.AvgPool2d((1, pool_kernel), (1, pool_stride)),
            nn.Dropout(dropout),
        )

        # proekciya tokenov v emb_dim (LazyLinear -> ne nado schitat' razmer vruchnuyu)
        self.proj = nn.LazyLinear(emb_dim)

        self.cls = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.pos = PositionalEncoding(emb_dim)
        self.subject = nn.Embedding(n_subjects, emb_dim) if n_subjects > 1 else None

        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor, subject_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, T) -> (B, 1, C, T)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = self.tokenizer(x.unsqueeze(1))          # (B, F, 1, T')
        b, f, _, t = h.shape
        h = h.reshape(b, f, t).transpose(1, 2)      # (B, T', F)
        h = self.proj(h)                            # (B, T', emb)

        if self.subject is not None and subject_id is not None:
            h = h + self.subject(subject_id).unsqueeze(1)

        cls = self.cls.expand(b, -1, -1)            # (B, 1, emb)
        h = torch.cat([cls, h], dim=1)              # (B, 1+T', emb)
        h = self.pos(h)
        h = self.encoder(h)
        return self.norm(h[:, 0])                   # CLS -> (B, emb)
