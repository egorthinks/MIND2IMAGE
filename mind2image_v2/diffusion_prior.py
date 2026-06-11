"""Diffusion Prior dlya MIND2IMAGE 2.0.

Po motivam DALLE-2 prior i MindEye2: malen'kaya diffuzionnaya set', kotoraya iz
brain-embedding vosstanavlivaet *raspredelenie* istinnyh CLIP image embeddingov.
Eto zametno podnimaet vernost' rekonstrukcii po sravneniyu s pryamoy MSE-regressiey
v alignment.AlignmentHead.

Diffuziya idyot v prostranstve embeddingov (1D-vektor D=768), a ne v pikselyah —
poetomu set' lyogkaya i bystro uchitsya.
"""

from __future__ import annotations

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal'noe kodirovanie shaga diffuzii. t: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class PriorDenoiser(nn.Module):
    """Transformer-denoiser: predskazyvaet chistyy CLIP-emb iz zashumlyonnogo,
    obuslovlennyy brain-embeddingom i shagom t."""

    def __init__(self, dim: int = 768, depth: int = 6, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            batch_first=True, activation="gelu",
        )
        self.net = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = nn.Linear(dim, dim)

    def forward(self, noisy: torch.Tensor, t: torch.Tensor, brain: torch.Tensor) -> torch.Tensor:
        # tri tokena: [time, brain (uslovie), noisy (chto chistim)]
        te = self.time_mlp(_timestep_embedding(t, self.dim))
        tokens = torch.stack([te, brain, noisy], dim=1)   # (B, 3, D)
        h = self.net(tokens)
        return self.out(h[:, -1])                          # predskazannyy chistyy CLIP-emb


class GaussianDiffusion1D:
    """Prostoy DDPM v prostranstve embeddingov (lineynyy beta-grafik, x0-predskazanie)."""

    def __init__(self, timesteps: int = 1000, device: str = "cpu"):
        self.T = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.device = device

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t].view(-1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def p_loss(self, model: PriorDenoiser, x0: torch.Tensor, brain: torch.Tensor) -> torch.Tensor:
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred_x0 = model(xt, t, brain)
        return F.mse_loss(pred_x0, x0)

    @torch.no_grad()
    def sample(self, model: PriorDenoiser, brain: torch.Tensor) -> torch.Tensor:
        """DDIM-podobnyy sampling: iz shuma -> CLIP-emb, obuslovlennyy brain."""
        x = torch.randn_like(brain)
        for i in reversed(range(self.T)):
            t = torch.full((brain.size(0),), i, device=brain.device, dtype=torch.long)
            pred_x0 = model(x, t, brain)
            ab = self.alpha_bar[i]
            if i > 0:
                ab_prev = self.alpha_bar[i - 1]
                # determinirovannyy DDIM-shag
                noise_dir = (x - ab.sqrt() * pred_x0) / (1 - ab).sqrt()
                x = ab_prev.sqrt() * pred_x0 + (1 - ab_prev).sqrt() * noise_dir
            else:
                x = pred_x0
        return F.normalize(x, dim=-1)


class DiffusionPrior(nn.Module):
    """Udobnaya obyortka: denoiser + raspisanie."""

    def __init__(self, dim: int = 768, depth: int = 6, heads: int = 8, timesteps: int = 1000):
        super().__init__()
        self.denoiser = PriorDenoiser(dim, depth, heads)
        self.timesteps = timesteps
        self._diff: Optional[GaussianDiffusion1D] = None

    def _diffusion(self, device) -> GaussianDiffusion1D:
        if self._diff is None or self._diff.device != device:
            self._diff = GaussianDiffusion1D(self.timesteps, device=device)
        return self._diff

    def loss(self, clip_emb: torch.Tensor, brain_emb: torch.Tensor) -> torch.Tensor:
        return self._diffusion(clip_emb.device).p_loss(self.denoiser, clip_emb, brain_emb)

    @torch.no_grad()
    def predict(self, brain_emb: torch.Tensor) -> torch.Tensor:
        return self._diffusion(brain_emb.device).sample(self.denoiser, brain_emb)
