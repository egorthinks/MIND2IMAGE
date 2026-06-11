"""Masked EEG Modeling — self-supervised pretrain dlya EEG Conformer.

Ideya (kak LaBraM / EEG-MAE / BENDR): maskируем sluchaynye vremennye uchastki
EEG i uchim model' vosstanavlivat' ih. Eto glavnyy apgreyd protiv potolka malyh
EEG-datasetov — encoder uchitsya na vsekh dostupnyh (v t.ch. nerazmechennyh) EEG
do togo, kak yego dootuchat' na parah EEG-kartinka.

Etap predobucheniya ne trebuet kartinok i CLIP — tol'ko sygnaly EEG.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn

from .encoders import EEGConformer


class MaskedEEGPretrainer(nn.Module):
    """Obyortka nad EEGConformer s rekonstrukcionnoy golovoy.

    Vmesto CLS-pulinga ispol'zuet vse vykhodnye tokeny encodera (cherez
    vnutrenniy dostup) — poetomu zdes' my dublируem token-uroven' rekonstrukcii
    cherez otdel'nuyu lyogkuyu golovu po zamaskirovannomu vkhodu.
    """

    def __init__(self, encoder: EEGConformer, seq_len: int,
                 mask_ratio: float = 0.5, mask_span: int = 20):
        super().__init__()
        self.encoder = encoder
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.mask_span = mask_span
        self.mask_token = nn.Parameter(torch.randn(encoder.in_channels) * 0.02)
        # golova rekonstrukcii: iz brain-embedding obratno v (C, T) grubo
        self.decoder = nn.Sequential(
            nn.Linear(encoder.emb_dim, encoder.emb_dim),
            nn.GELU(),
        )
        # golova zaregistrirovana srazu — chtoby yeyo parametry popali v optimizer
        self.out = nn.Linear(encoder.emb_dim, encoder.in_channels * seq_len)

    def _apply_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maskируем span'y po vremeni. x: (B, C, T). Vozvr. (masked_x, mask_bool[T])."""
        b, c, t = x.shape
        mask = torch.zeros(b, t, dtype=torch.bool, device=x.device)
        n_spans = max(1, int((t * self.mask_ratio) / self.mask_span))
        for i in range(b):
            for _ in range(n_spans):
                start = torch.randint(0, max(1, t - self.mask_span), (1,)).item()
                mask[i, start : start + self.mask_span] = True
        masked_x = x.clone()
        # zamena zamaskirovannyh otschyotov na mask_token (po kazhdomu kanalu)
        mt = self.mask_token.view(1, c, 1)
        masked_x = torch.where(mask.unsqueeze(1), mt.expand_as(x), masked_x)
        return masked_x, mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vozvrashchaet (loss, brain_embedding). x: (B, C, T)."""
        masked_x, mask = self._apply_mask(x)
        emb = self.encoder(masked_x)                 # (B, emb)
        h = self.decoder(emb)                        # (B, emb)
        recon = self.out(h).view_as(x)               # (B, C, T)
        # loss tol'ko po zamaskirovannym otschyotam
        m = mask.unsqueeze(1).expand_as(x)
        loss = ((recon - x) ** 2 * m).sum() / (m.sum() + 1e-7)
        return loss, emb
