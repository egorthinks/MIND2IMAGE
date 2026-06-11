"""Konfiguratsiya MIND2IMAGE 2.0."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Mind2ImageV2Config:
    # --- EEG vkhod ---
    in_channels: int = 128          # chislo EEG-kanalov
    seq_len: int = 440              # chislo otschyotov vo vremeni
    sample_rate: int = 1000         # Gts, dlya preprocessing

    # --- Preprocessing ---
    bandpass: Tuple[float, float] = (0.5, 45.0)
    notch: float = 50.0
    use_ica: bool = False           # trebует mne; po umolchaniyu off
    per_channel_zscore: bool = True

    # --- Encoder (Conformer) ---
    emb_dim: int = 768              # = CLIP ViT-L/14 image embedding dim
    conformer_depth: int = 6
    conformer_heads: int = 8
    conformer_temporal_kernel: int = 25
    conformer_pool_kernel: int = 15
    conformer_pool_stride: int = 5
    conformer_n_filters: int = 40
    dropout: float = 0.3
    n_subjects: int = 1             # subject-embedding dlya mezhsub"ektnoy obobshchaemosti

    # --- CLIP alignment ---
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    clip_dim: int = 768
    align_text_too: bool = False    # vyravnivat' i v text-embedding (esli est' captions)

    # --- Diffusion Prior ---
    prior_depth: int = 6
    prior_heads: int = 8
    prior_timesteps: int = 1000

    # --- Generaciya (Stable Diffusion unCLIP) ---
    sd_model: str = "stabilityai/stable-diffusion-2-1-unclip"
    guidance_scale: float = 10.0
    num_inference_steps: int = 30

    # --- Obuchenie ---
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    epochs: int = 100
    logit_scale_init: float = 2.6592   # = log(1/0.07), kak v CLIP
    logit_scale_max: float = 4.6052    # = log(100), klipping

    # vesa loss-ov
    w_contrastive: float = 1.0
    w_regression: float = 1.0
    mixco_alpha: float = 0.15

    metrics: List[str] = field(default_factory=lambda: [
        "clip_sim", "two_way_acc", "ssim", "retrieval_top1",
    ])
