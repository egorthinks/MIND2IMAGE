"""Smoke-test torch-yadra MIND2IMAGE 2.0 (CPU, sluchaynye tenzory).

Proveryaet, chto vsyo bez vneshnih zavisimostey (open_clip/diffusers/mne) rabotaet:
preprocessing, EEGConformer, AlignmentHead, losses, DiffusionPrior, MaskedEEGPretrainer.
Zapusk:  python -m mind2image_v2.smoke_test
"""

import numpy as np
import torch

from mind2image_v2.config import Mind2ImageV2Config
from mind2image_v2.preprocessing import preprocess_batch
from mind2image_v2.pipeline import Mind2ImageV2
from mind2image_v2.masked_pretrain import MaskedEEGPretrainer
from mind2image_v2.losses import (
    clip_contrastive_loss, soft_clip_loss, mixco, mixco_contrastive_loss, regression_loss,
)


def main():
    torch.manual_seed(0)
    B, C, T = 4, 32, 256                     # malen'kie razmery dlya CPU
    cfg = Mind2ImageV2Config(in_channels=C, seq_len=T, emb_dim=128, clip_dim=128,
                             conformer_depth=2, prior_depth=2, prior_timesteps=20,
                             n_subjects=3)

    # 1) preprocessing (numpy fallback, bez scipy/mne)
    raw = np.random.randn(B, C, T).astype(np.float32)
    pre = preprocess_batch(raw, fs=cfg.sample_rate, bandpass=cfg.bandpass, notch=cfg.notch)
    assert pre.shape == (B, C, T)
    # pokanal'nyy z-score: srednее ~0, std ~1
    assert abs(float(pre.mean())) < 0.1 and abs(float(pre.std()) - 1.0) < 0.2
    print(f"[1] preprocessing OK  shape={pre.shape} mean={pre.mean():.3f} std={pre.std():.3f}")

    eeg = torch.from_numpy(pre)
    subj = torch.randint(0, cfg.n_subjects, (B,))

    # 2) model + encoder + alignment
    model = Mind2ImageV2(cfg)
    brain = model.encoder(eeg, subj)
    assert brain.shape == (B, cfg.emb_dim), brain.shape
    aligned = model.encode(eeg, subj)
    assert aligned.shape == (B, cfg.clip_dim), aligned.shape
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] encoder->align OK  brain={tuple(brain.shape)} aligned={tuple(aligned.shape)} "
          f"params={n_params/1e6:.2f}M")

    # 3) losses (s sintetic CLIP-targetom)
    target = torch.nn.functional.normalize(torch.randn(B, cfg.clip_dim), dim=-1)
    scale = model.align.scale()
    assert float(scale.detach()) <= np.exp(cfg.logit_scale_max) + 1
    l_clip = clip_contrastive_loss(aligned, target, scale)
    l_soft = soft_clip_loss(aligned, target, scale)
    mixed, perm, lam = mixco(brain, cfg.mixco_alpha)
    l_mix = mixco_contrastive_loss(model.align(mixed), target, perm, lam, scale)
    l_reg = regression_loss(aligned, target)
    total = l_clip + l_soft + l_mix + l_reg
    print(f"[3] losses OK  clip={l_clip:.3f} soft={l_soft:.3f} mix={l_mix:.3f} reg={l_reg:.3f}")

    # 3b) backward dohodit do encodera, align i obuchaemogo logit_scale
    total.backward()
    assert model.align.logit_scale.grad is not None, "logit_scale ne obuchaetsya!"
    assert any(p.grad is not None for p in model.encoder.parameters())
    print(f"[3b] backward OK  logit_scale.grad={float(model.align.logit_scale.grad):.4f}")

    # 4) diffusion prior: loss + sampling
    model.zero_grad()
    p_loss = model.prior.loss(target, aligned.detach())
    p_loss.backward()
    assert any(p.grad is not None for p in model.prior.parameters())
    sampled = model.prior.predict(aligned.detach())
    assert sampled.shape == (B, cfg.clip_dim), sampled.shape
    print(f"[4] diffusion prior OK  loss={p_loss:.3f} sample={tuple(sampled.shape)} "
          f"|sample|={sampled.norm(dim=-1).mean():.3f}")

    # 5) masked pretrain: golova reconstruction popadaet v optimizer i grad teper'
    pre_m = MaskedEEGPretrainer(model.encoder, seq_len=T, mask_ratio=0.5, mask_span=16)
    opt = torch.optim.AdamW(pre_m.parameters(), lr=1e-3)
    out_params_before = pre_m.out.weight.detach().clone()
    loss_m, emb_m = pre_m(eeg)
    opt.zero_grad(); loss_m.backward(); opt.step()
    assert pre_m.out.weight.grad is not None, "out-golova ne v optimizer!"
    moved = (pre_m.out.weight.detach() - out_params_before).abs().sum().item()
    assert moved > 0, "out-golova ne obnovilas'"
    print(f"[5] masked pretrain OK  loss={loss_m:.3f} emb={tuple(emb_m.shape)} "
          f"out_update={moved:.4f}")

    print("\nALL SMOKE TESTS PASSED ✔")


if __name__ == "__main__":
    main()
