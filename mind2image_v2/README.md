# mind2image_v2 — реализация архитектуры 2.0

Параллельная реализация пайплайна из [`../ARCHITECTURE_2.0.md`](../ARCHITECTURE_2.0.md).
**Старый код 1.0 (`models/`, `train.py`, `generate.py`) не тронут** — это отдельный пакет.

```
EEG ─▶ preprocessing ─▶ EEG Conformer ─▶ AlignmentHead(CLIP) ─▶ DiffusionPrior ─▶ Stable Diffusion ─▶ image
        (preprocessing)   (encoders)        (alignment)          (diffusion_prior)   (generation)
```

## Карта модулей

| Файл | Что делает | Заменяет в 1.0 |
|------|-----------|----------------|
| `preprocessing.py` | bandpass + notch + (опц.) ICA + поканальный z-score | `(eeg-max/2)/(max/2)` в `models/dataset.py` |
| `encoders.py` | `EEGConformer` (conv-токенизатор + Transformer + CLS + pos-enc + subject-emb) | `EEGLSTM_Encoder`/`EEGTransformer`/`EEG_CNN` |
| `masked_pretrain.py` | Masked-EEG self-supervised претрейн | — (нового нет в 1.0) |
| `alignment.py` | `AlignmentHead` (brain→CLIP) + `CLIPTarget`; обучаемый `logit_scale` | `Image_encoder` (ResNet50) + битая temperature |
| `losses.py` | SoftCLIP, BiMixCo, регрессия cos+MSE | сломанный InfoNCE в `train.py` |
| `diffusion_prior.py` | `DiffusionPrior` (DALLE-2/MindEye2 prior) | — |
| `generation.py` | `StableDiffusionGenerator` (SD unCLIP) | StyleGAN2-ADA |
| `pipeline.py` | `Mind2ImageV2` — сшивка всего | — |
| `dataset.py` | сырые EEG+image+subject_id | `models/dataset.py` |
| `train.py` | стадии A/B/C | `FeatureExtractorUser`/`ADAUser` |
| `config.py` | `Mind2ImageV2Config` | `cfg.py` |

## Зависимости

Torch-ядро (`encoders`, `diffusion_prior`, `losses`, `pipeline`, `alignment.AlignmentHead`)
работает только на `torch`. Остальное подгружается **лениво** — пакет импортируется
без них:

- `open_clip_torch` — для `CLIPTarget` (целевые CLIP-эмбеддинги, стадии B/C);
- `diffusers transformers accelerate` — для `StableDiffusionGenerator` (инференс);
- `mne` / `scipy` — для ICA и фильтров (есть numpy-fallback);
- `opencv-python` — для `dataset.py`.

```bash
pip install torch open_clip_torch diffusers transformers accelerate scipy mne opencv-python
```

## Пример использования

```python
import torch
from mind2image_v2 import Mind2ImageV2, Mind2ImageV2Config

cfg = Mind2ImageV2Config(in_channels=128, seq_len=440, n_subjects=6)
model = Mind2ImageV2(cfg)

# forward energcheskogo yadra (bez vneshnih zavisimostey):
eeg = torch.randn(4, 128, 440)
brain_clip = model.encode(eeg)          # (4, 768) v CLIP-prostranstve
prior_loss = model.prior.loss(brain_clip.detach(), brain_clip.detach())

# полный инференс (нужны open_clip + diffusers):
# images = model.reconstruct(eeg, device="cuda")
```

## Стадии обучения

```python
from torch.utils.data import DataLoader
from mind2image_v2.dataset import EEGOnlyDataset, EEGImageDatasetV2
from mind2image_v2.train import pretrain_masked, train_alignment, train_prior

# A) self-supervised претрейн энкодера (только EEG)
ssl_loader = DataLoader(EEGOnlyDataset("data/*.npy"), batch_size=64, shuffle=True)
model.encoder = pretrain_masked(model.encoder, ssl_loader, seq_len=cfg.seq_len, epochs=50)

# B) выравнивание в CLIP
pair_loader = DataLoader(EEGImageDatasetV2("data/*.npy"), batch_size=32, shuffle=True)
model = train_alignment(model, pair_loader, cfg)

# C) диффузионный prior
model = train_prior(model, pair_loader, cfg)
```

Стадии A и C дают наибольший прирост и независимы — можно начинать с любой.
