"""Sквoznoy pipeline MIND2IMAGE 2.0.

Svyazyvaet: EEG Conformer -> AlignmentHead -> DiffusionPrior -> Stable Diffusion.

Torch-yadro (encoder + head + prior) rabotaet bez vneshnih zavisimostey.
Generaciya (Stable Diffusion) i CLIP-targety podklyuchayutsya lenivo.
"""

from __future__ import annotations

from typing import Optional, List
import torch
import torch.nn as nn

from .config import Mind2ImageV2Config
from .encoders import EEGConformer
from .alignment import AlignmentHead, CLIPTarget
from .diffusion_prior import DiffusionPrior


class Mind2ImageV2(nn.Module):
    """End-to-end model' (bez generatora — on podklyuchaetsya otdel'no pri inference)."""

    def __init__(self, cfg: Optional[Mind2ImageV2Config] = None):
        super().__init__()
        self.cfg = cfg or Mind2ImageV2Config()
        c = self.cfg

        self.encoder = EEGConformer(
            in_channels=c.in_channels,
            emb_dim=c.emb_dim,
            depth=c.conformer_depth,
            heads=c.conformer_heads,
            temporal_kernel=c.conformer_temporal_kernel,
            pool_kernel=c.conformer_pool_kernel,
            pool_stride=c.conformer_pool_stride,
            n_filters=c.conformer_n_filters,
            dropout=c.dropout,
            n_subjects=c.n_subjects,
        )
        self.align = AlignmentHead(
            in_dim=c.emb_dim,
            clip_dim=c.clip_dim,
            logit_scale_init=c.logit_scale_init,
            logit_scale_max=c.logit_scale_max,
        )
        self.prior = DiffusionPrior(
            dim=c.clip_dim,
            depth=c.prior_depth,
            heads=c.prior_heads,
            timesteps=c.prior_timesteps,
        )

        # lenivye komponenty
        self._clip_target: Optional[CLIPTarget] = None
        self._generator = None

    # ---- encoder + alignment ----
    def encode(self, eeg: torch.Tensor, subject_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """eeg: (B, C, T) -> brain embedding v CLIP-prostranstve (B, clip_dim)."""
        brain = self.encoder(eeg, subject_id)
        return self.align(brain)

    # ---- lenivye akcessory ----
    def clip_target(self, device: str = "cuda") -> CLIPTarget:
        if self._clip_target is None:
            self._clip_target = CLIPTarget(
                self.cfg.clip_model, self.cfg.clip_pretrained, device
            )
        return self._clip_target

    def generator(self, device: str = "cuda"):
        if self._generator is None:
            from .generation import StableDiffusionGenerator
            self._generator = StableDiffusionGenerator(self.cfg.sd_model, device)
        return self._generator

    # ---- inference: EEG -> kartinka ----
    @torch.no_grad()
    def reconstruct(self, eeg: torch.Tensor, subject_id: Optional[torch.Tensor] = None,
                    use_prior: bool = True, device: str = "cuda") -> List["object"]:
        """Polnaya rekonstrukciya EEG -> spisok PIL.Image."""
        self.eval()
        aligned = self.encode(eeg.to(device), subject_id)
        clip_emb = self.prior.predict(aligned) if use_prior else aligned
        return self.generator(device).generate(
            clip_emb,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
        )
