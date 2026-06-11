"""Vyravnivanie brain-embedding v prostranstvo CLIP.

V 1.0 EEG vyravnivalsya v samodel'noe prostranstvo ResNet50 (Image_encoder), kotoroe
ni s chem ne stykuetsya. V 2.0 cel'yu vyravnivaniya stanovitsya zamorozhennyy CLIP
ViT-L image embedding — togo zhe prostranstva zhdyot Stable Diffusion (unCLIP/IP-Adapter).

CLIPTarget — lenivaya obyortka nad open_clip (importiruetsya tol'ko pri vyzove).
AlignmentHead — obuchaemaya proekciya brain-emb -> CLIP-emb.
"""

from __future__ import annotations

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentHead(nn.Module):
    """Proekciya brain-embedding v CLIP-prostranstvo + obuchaemyy logit_scale."""

    def __init__(self, in_dim: int = 768, clip_dim: int = 768,
                 hidden: int = 2048, logit_scale_init: float = 2.6592,
                 logit_scale_max: float = 4.6052):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, clip_dim),
        )
        self.residual = nn.Linear(in_dim, clip_dim)
        # obuchaemaya temperatura (v 1.0 byla slomana: torch.exp(tensor(const)))
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))
        self.logit_scale_max = logit_scale_max

    def forward(self, brain_emb: torch.Tensor) -> torch.Tensor:
        out = self.mlp(brain_emb) + self.residual(brain_emb)
        return out

    def scale(self) -> torch.Tensor:
        return self.logit_scale.clamp(max=self.logit_scale_max).exp()


class CLIPTarget:
    """Lenivaya obyortka nad zamorozhennym CLIP dlya polucheniya celevyh embeddingov.

    Trebует `open_clip_torch`. Esli ne ustanovlen — podnimaet ponyatnuyu oshibku
    tol'ko v moment vyzova, a ne pri importe paketa.
    """

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai",
                 device: str = "cuda"):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _ensure(self):
        if self._model is not None:
            return
        try:
            import open_clip  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "CLIPTarget trebует 'open_clip_torch' (pip install open_clip_torch)"
            ) from e
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._model = self._model.eval().to(self.device)
        for p in self._model.parameters():
            p.requires_grad = False
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    @torch.no_grad()
    def image_embed(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) uzhe normalizovannye pod CLIP. -> (B, clip_dim)."""
        self._ensure()
        feat = self._model.encode_image(images.to(self.device))
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def text_embed(self, captions: List[str]) -> torch.Tensor:
        self._ensure()
        tok = self._tokenizer(captions).to(self.device)
        feat = self._model.encode_text(tok)
        return F.normalize(feat, dim=-1)

    @property
    def preprocess(self):
        self._ensure()
        return self._preprocess
