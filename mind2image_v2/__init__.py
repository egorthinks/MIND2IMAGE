"""MIND2IMAGE 2.0 — neinvazivnyy EEG -> Image BCI.

Novaya arkhitektura (sm. ARCHITECTURE_2.0.md):
    EEG -> preprocessing -> EEG Conformer (SSL pretrain)
        -> CLIP-alignment -> Diffusion Prior -> Stable Diffusion.

Staryy pipeline (StyleGAN2-ADA) ostayotsya netronutym v `models/`, `train.py`,
`generate.py`. Etot paket — parallel'naya realizatsiya 2.0.

Tyazhyolye zavisimosti (open_clip, diffusers, mne) importiruyutsya lenivo, poetomu
torch-yadro (encoders, diffusion_prior, losses, pipeline) rabotaet samostoyatel'no.
"""

from .config import Mind2ImageV2Config
from .encoders import EEGConformer
from .alignment import AlignmentHead, CLIPTarget
from .diffusion_prior import DiffusionPrior, GaussianDiffusion1D
from .losses import (
    clip_contrastive_loss,
    soft_clip_loss,
    mixco,
    regression_loss,
)
from .pipeline import Mind2ImageV2

__all__ = [
    "Mind2ImageV2Config",
    "EEGConformer",
    "AlignmentHead",
    "CLIPTarget",
    "DiffusionPrior",
    "GaussianDiffusion1D",
    "Mind2ImageV2",
    "clip_contrastive_loss",
    "soft_clip_loss",
    "mixco",
    "regression_loss",
]

__version__ = "2.0.0"
