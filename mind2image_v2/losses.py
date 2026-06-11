"""Loss-funkcii dlya vyravnivaniya brain <-> CLIP.

Pravil'naya zamena slomannogo InfoNCE iz 1.0 (train.py:144, gde temperatura byla
konstantoy `torch.exp(torch.tensor(temperature))`).

  * clip_contrastive_loss — simmetrichnyy InfoNCE s obuchaemym logit_scale;
  * soft_clip_loss — myagkie targety (MindEye SoftCLIP), ustoychiv k shumu EEG;
  * mixco — MixCo/BiMixCo augmentaciya v embedding-prostranstve;
  * regression_loss — cos + MSE prityagivaet k konkretnomu CLIP-embeddingu.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_contrastive_loss(brain: torch.Tensor, clip: torch.Tensor,
                          logit_scale: torch.Tensor) -> torch.Tensor:
    """Simmetrichnyy InfoNCE. brain, clip: (B, D) (zhelatel'no L2-normalizovany)."""
    brain = F.normalize(brain, dim=-1)
    clip = F.normalize(clip, dim=-1)
    logits = logit_scale * brain @ clip.t()        # (B, B)
    labels = torch.arange(brain.size(0), device=brain.device)
    loss_b = F.cross_entropy(logits, labels)
    loss_c = F.cross_entropy(logits.t(), labels)
    return (loss_b + loss_c) / 2.0


def soft_clip_loss(brain: torch.Tensor, clip: torch.Tensor,
                   logit_scale: torch.Tensor) -> torch.Tensor:
    """SoftCLIP: targety = vnutren'ee shodstvo clip<->clip (myagkie metki).

    Pomogaet pri shumnyh EEG, gde zhyostkaya diagonal' slishkom optimistichna.
    """
    brain = F.normalize(brain, dim=-1)
    clip = F.normalize(clip, dim=-1)
    scale = logit_scale
    logits = scale * brain @ clip.t()
    with torch.no_grad():
        targets = F.softmax(scale * clip @ clip.t(), dim=-1)
    loss_b = -(targets * F.log_softmax(logits, dim=-1)).sum(1).mean()
    loss_c = -(targets.t() * F.log_softmax(logits.t(), dim=-1)).sum(1).mean()
    return (loss_b + loss_c) / 2.0


def mixco(brain: torch.Tensor, alpha: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MixCo: lineynoe smeshivanie brain-embeddingov dlya regularizatsii.

    Vozvrashchaet (smeshannyy_brain, perm_indices, lam).
    """
    b = brain.size(0)
    perm = torch.randperm(b, device=brain.device)
    lam = torch.distributions.Beta(alpha, alpha).sample((b,)).to(brain.device)
    lam = lam.view(-1, 1)
    mixed = lam * brain + (1 - lam) * brain[perm]
    return mixed, perm, lam.squeeze(1)


def mixco_contrastive_loss(mixed_brain: torch.Tensor, clip: torch.Tensor,
                           perm: torch.Tensor, lam: torch.Tensor,
                           logit_scale: torch.Tensor) -> torch.Tensor:
    """InfoNCE dlya smeshannyh embeddingov (BiMixCo)."""
    mixed_brain = F.normalize(mixed_brain, dim=-1)
    clip = F.normalize(clip, dim=-1)
    logits = logit_scale * mixed_brain @ clip.t()
    n = mixed_brain.size(0)
    idx = torch.arange(n, device=mixed_brain.device)
    logp = F.log_softmax(logits, dim=-1)
    loss = -(lam * logp[idx, idx] + (1 - lam) * logp[idx, perm]).mean()
    return loss


def regression_loss(brain: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
    """Cos + MSE: prityagivaet predskazanie k konkretnomu CLIP-embeddingu."""
    cos = 1 - F.cosine_similarity(brain, clip, dim=-1).mean()
    mse = F.mse_loss(F.normalize(brain, dim=-1), F.normalize(clip, dim=-1))
    return cos + mse
