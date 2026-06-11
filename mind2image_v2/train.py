"""Trenirovochnye stadii MIND2IMAGE 2.0.

Tri nezavisimye stadii (sm. ARCHITECTURE_2.0.md, razdel 6):
  A) pretrain_masked    — self-supervised pretrain encodera na syryh EEG;
  B) train_alignment    — vyravnivanie brain-emb v CLIP-prostranstvo;
  C) train_prior        — diffuzionnyy prior brain-emb -> CLIP-emb.

Kazhduyu mozhno zapuskat' otdel'no. CLIP-targety (stadiya B) trebuyut open_clip.
"""

from __future__ import annotations

from typing import Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Mind2ImageV2Config
from .pipeline import Mind2ImageV2
from .encoders import EEGConformer
from .masked_pretrain import MaskedEEGPretrainer
from .losses import (
    clip_contrastive_loss, soft_clip_loss, mixco, mixco_contrastive_loss, regression_loss,
)


# ---------- Stadiya A: self-supervised pretrain ----------
def pretrain_masked(encoder: EEGConformer, loader: DataLoader, seq_len: int,
                    epochs: int = 50, lr: float = 3e-4, device: str = "cuda",
                    mask_ratio: float = 0.5) -> EEGConformer:
    pre = MaskedEEGPretrainer(encoder, seq_len=seq_len, mask_ratio=mask_ratio).to(device)
    opt = torch.optim.AdamW(pre.parameters(), lr=lr, weight_decay=1e-2)
    for ep in range(epochs):
        pre.train()
        run = 0.0
        for eeg in tqdm(loader, desc=f"[SSL] epoch {ep+1}/{epochs}"):
            eeg = eeg.to(device)
            opt.zero_grad()
            loss, _ = pre(eeg)
            loss.backward()
            opt.step()
            run += loss.item()
        print(f"[SSL] epoch {ep+1}: loss={run/len(loader):.4f}")
    return pre.encoder


# ---------- Stadiya B: vyravnivanie v CLIP ----------
def train_alignment(model: Mind2ImageV2, loader: DataLoader,
                    cfg: Optional[Mind2ImageV2Config] = None, device: str = "cuda"):
    cfg = cfg or model.cfg
    model = model.to(device)
    clip = model.clip_target(device)
    # CLIP-preprocessing dlya tenzorov kartinok
    import torchvision.transforms as T
    clip_norm = T.Normalize((0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711))

    params = list(model.encoder.parameters()) + list(model.align.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    for ep in range(cfg.epochs):
        model.train()
        run = 0.0
        for eeg, img, _, subj in tqdm(loader, desc=f"[ALIGN] epoch {ep+1}/{cfg.epochs}"):
            eeg, img, subj = eeg.to(device), img.to(device), subj.to(device)
            opt.zero_grad()

            # celevoy CLIP image embedding (zamorozhennyy)
            img_clip = torch.nn.functional.interpolate(img, size=224, mode="bilinear",
                                                       align_corners=False)
            target = clip.image_embed(clip_norm(img_clip))

            brain = model.encode(eeg, subj)
            scale = model.align.scale()

            # BiMixCo + SoftCLIP + regressiya
            mixed, perm, lam = mixco(model.encoder(eeg, subj), cfg.mixco_alpha)
            mixed = model.align(mixed)
            loss = (
                cfg.w_contrastive * soft_clip_loss(brain, target, scale)
                + cfg.w_contrastive * mixco_contrastive_loss(mixed, target, perm, lam, scale)
                + cfg.w_regression * regression_loss(brain, target)
            )
            loss.backward()
            opt.step()
            run += loss.item()
        print(f"[ALIGN] epoch {ep+1}: loss={run/len(loader):.4f}")
    return model


# ---------- Stadiya C: diffuzionnyy prior ----------
def train_prior(model: Mind2ImageV2, loader: DataLoader,
                cfg: Optional[Mind2ImageV2Config] = None, device: str = "cuda"):
    cfg = cfg or model.cfg
    model = model.to(device)
    clip = model.clip_target(device)
    import torchvision.transforms as T
    clip_norm = T.Normalize((0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711))

    # encoder/align zamorozheny — uchim tol'ko prior
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.align.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.prior.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for ep in range(cfg.epochs):
        model.prior.train()
        run = 0.0
        for eeg, img, _, subj in tqdm(loader, desc=f"[PRIOR] epoch {ep+1}/{cfg.epochs}"):
            eeg, img, subj = eeg.to(device), img.to(device), subj.to(device)
            opt.zero_grad()
            with torch.no_grad():
                img_clip = torch.nn.functional.interpolate(img, size=224, mode="bilinear",
                                                           align_corners=False)
                target = clip.image_embed(clip_norm(img_clip))
                brain = model.encode(eeg, subj)
            loss = model.prior.loss(target, brain)
            loss.backward()
            opt.step()
            run += loss.item()
        print(f"[PRIOR] epoch {ep+1}: loss={run/len(loader):.4f}")
    return model
