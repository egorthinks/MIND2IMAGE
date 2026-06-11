"""Generaciya kartinok iz predskazannogo CLIP-embedding cherez Stable Diffusion.

Zamena StyleGAN2-ADA iz 1.0. Ispol'zuetsya zamorozhennaya SD unCLIP
(stabilityai/stable-diffusion-2-1-unclip), kotoraya prinimaet imenno CLIP image
embedding kak uslovie cherez cross-attention. Otkrytyy domen, fotorealizm i
stabil'nost' "besplatno" — generator ne uchitsya s nulya pod mozg.

`diffusers` importiruetsya lenivo: paket mozhno importirovat' bez nego.
"""

from __future__ import annotations

from typing import List, Optional
import torch


class StableDiffusionGenerator:
    """Lenivaya obyortka nad SD unCLIP pipeline."""

    def __init__(self, model: str = "stabilityai/stable-diffusion-2-1-unclip",
                 device: str = "cuda", dtype: Optional["torch.dtype"] = None):
        self.model = model
        self.device = device
        self.dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return
        try:
            from diffusers import StableUnCLIPImg2ImgPipeline  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "StableDiffusionGenerator trebует 'diffusers' (pip install diffusers transformers accelerate)"
            ) from e
        self._pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            self.model, torch_dtype=self.dtype
        ).to(self.device)

    @torch.no_grad()
    def generate(self, clip_image_embeds: torch.Tensor,
                 num_inference_steps: int = 30,
                 guidance_scale: float = 10.0,
                 prompt: str = "") -> List["object"]:
        """clip_image_embeds: (B, clip_dim) -> spisok PIL.Image."""
        self._ensure()
        embeds = clip_image_embeds.to(self.device, self.dtype)
        images = []
        for i in range(embeds.size(0)):
            out = self._pipe(
                image_embeds=embeds[i : i + 1],
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            images.append(out.images[0])
        return images
