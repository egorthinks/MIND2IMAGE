# MIND2IMAGE 2.0 — архитектура неинвазивного EEG→Image BCI

Документ: разбор текущей архитектуры (1.0), её узких мест и предложение
архитектуры 2.0 на современных, но **подходящих именно для EEG** компонентах.

> Главная мысль: проблема EEG-to-image — это не «слабый генератор», а **низкий
> SNR и малые датасеты**. Поэтому ставка 2.0 не на «более модный GAN», а на
> (1) самообучение энкодера на сырых EEG, (2) выравнивание в готовое CLIP-пространство
> и (3) генерацию диффузией с огромным предобученным приором. Генератор перестаёт
> учиться «с нуля» под мозг — он уже умеет рисовать мир.

---

## 1. Что есть сейчас (1.0)

| Блок | Реализация | Файл |
|------|-----------|------|
| EEG-энкодер | LSTM / Transformer / CNN / ResNet → 256-d | `models/model.py` |
| Image-энкодер | замороженный ResNet50 + MLP-голова → 256-d | `models/model.py: Image_encoder` |
| Выравнивание | InfoNCE (CLIP-style) EEG↔image | `train.py: FeatureExtractorUser.train` |
| Генерация | StyleGAN2-ADA, EEG-эмбеддинг как условие `c` | `generate.py`, `train.py: ADAUser` |
| Препроцессинг | `(eeg - max/2)/(max/2)` | `models/dataset.py` |

Пайплайн: `EEG → энкодер → 256-d эмбеддинг → conditional StyleGAN2-ADA → картинка`.
По сути это уровень Brain2Image / ThoughtViz (2017–2022): CLIP-энкодер + conditional GAN.

### Конкретные технические проблемы в коде

1. **GAN как генератор.** StyleGAN2-ADA учится с нуля и привязан к узкому домену
   (один датасет, например 40 классов). Нет открытого домена, нет семантического приора,
   тяжёлая и нестабильная тренировка.
2. **Несостыковка условия.** Генератор объявлен с `z_dim=512, w_dim=512`
   (`train.py:319`), а EEG-эмбеддинг 256-d. Подача `c` как классового условия не несёт
   богатой семантики.
3. **Image-энкодер = ResNet50, а не CLIP.** Выравнивание идёт в «самодельное»
   пространство, которое потом ни с чем не стыкуется. Современные генераторы ждут
   именно CLIP-эмбеддинги.
4. **Temperature сломана.** `torch.exp(torch.tensor(temperature))` (`train.py:144`) —
   константа, а не обучаемый `logit_scale`. В CLIP это обучаемый параметр с клиппингом.
5. **EEGTransformer без позиционного кодирования** и берёт последний токен
   (`model.py:140`) вместо CLS/pooling — для последовательности это слабо.
6. **Препроцессинг наивный.** Нормировка по глобальному максимуму, без полосовой
   фильтрации, ICA/удаления артефактов, без поканальной стандартизации, без baseline.
7. **Нет self-supervised претрейна** — энкодер учится только на парах EEG–картинка,
   которых мало. Это главный потолок качества.

---

## 2. Что изменилось в SOTA (и что из этого *подходит* EEG)

Беру только то, что реально работает на неинвазивном, малоданном, шумном EEG —
без fMRI-only трюков и без инвазива.

- **Диффузия вместо GAN.** Latent Diffusion / Stable Diffusion как замороженный
  генератор (DreamDiffusion — это прямо EEG→image; MindEye/MindEye2, Brain-Diffuser
  для fMRI). Открытый домен, стабильная тренировка, фотореализм «бесплатно».
- **Выравнивание в CLIP-пространство.** Учим EEG-эмбеддинг предсказывать **CLIP ViT-L
  image embedding** (и опционально text/caption embedding). Это даёт прямой вход в
  unCLIP/IP-Adapter и переносит на нас весь зрительный приор CLIP.
- **Диффузионный приор (DALLE-2 / MindEye2).** Маленькая сеть, которая из «мозгового»
  эмбеддинга восстанавливает *распределение* истинных CLIP-эмбеддингов. Резко поднимает
  верность реконструкции по сравнению с прямой MSE-регрессией.
- **Self-supervised foundation для EEG.** Masked EEG Modeling / нейротокенизатор
  (LaBraM — Large Brain Model, 2024; BENDR; EEG-MAE). Претрейн на сырых неразмеченных
  EEG решает проблему малых датасетов — это самый важный апгрейд после диффузии.
- **EEG Conformer как энкодер.** Свёрточный токенизатор (пространство+время) + трансформер.
  SOTA для декодирования EEG, лучше голого LSTM/Transformer.
- **Двухпотоковая реконструкция (low+high level).** Brain-Diffuser/MindEye: один поток
  даёт грубую структуру (initial latent / VDVAE), второй — семантику (CLIP). Склейка
  через img2img в SD.
- **Учёт межсубъектной вариативности.** Subject-эмбеддинги / shared-subject адаптеры
  (MindEye2), чтобы модель не разваливалась между испытуемыми.

---

## 3. MIND2IMAGE 2.0 — целевая архитектура

```
                    ┌─────────────────────────────────────────────┐
   raw EEG  ─────▶  │ 0. Препроцессинг: bandpass + notch + ICA-арт.│
 (C×T, напр.        │    поканальная стандартизация, baseline      │
  128×440)          └──────────────────┬──────────────────────────┘
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │ 1. EEG Foundation Encoder (Conformer)        │
                    │    нейротокенизатор (conv spatial+temporal)  │
                    │    + Transformer + subject-embedding         │
                    │    ── претрейн: Masked EEG Modeling (SSL) ── │
                    └──────────────────┬──────────────────────────┘
                                       ▼  brain embedding (b)
                    ┌─────────────────────────────────────────────┐
                    │ 2. Alignment head → CLIP space               │
                    │    проекция b → ĉ_img (768/1280-d)           │
                    │    loss: SoftCLIP/InfoNCE + cos/MSE          │
                    │    цель: замороженный CLIP ViT-L image emb   │
                    └──────────────────┬──────────────────────────┘
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │ 3. Diffusion Prior (DALLE-2/MindEye2)        │
                    │    ĉ_img → p(c_clip | brain)                 │
                    └──────────────────┬──────────────────────────┘
                                       ▼  предсказанный CLIP-emb
                    ┌─────────────────────────────────────────────┐
                    │ 4. Генератор: замороженная Stable Diffusion  │
                    │    unCLIP / IP-Adapter cross-attention       │
                    │    (опц.) low-level поток → img2img init     │
                    └──────────────────┬──────────────────────────┘
                                       ▼
                                  реконструкция
```

### Соответствие старое → новое

| 1.0 | 2.0 | Зачем |
|-----|-----|-------|
| LSTM/Transformer/CNN энкодер | **EEG Conformer + Masked-EEG претрейн** | SSL снимает потолок малых данных |
| ResNet50 image-энкодер | **замороженный CLIP ViT-L** | стыкуется с диффузией, богатая семантика |
| InfoNCE с битой temperature | **SoftCLIP + MSE/cos, обучаемый logit_scale** | стабильнее, регрессия в CLIP-emb |
| — | **Diffusion Prior** | верность реконструкции |
| StyleGAN2-ADA | **замороженная Stable Diffusion (unCLIP/IP-Adapter)** | открытый домен, фотореализм, стабильность |
| нормировка по max | **bandpass+notch+ICA, поканальная стандартизация** | реальный EEG-препроцессинг |
| один испытуемый | **subject-эмбеддинги** | межсубъектная переносимость |

---

## 4. Лоссы

- **Контрастив:** SoftCLIP/BiMixCo (InfoNCE с mixup и мягкими таргетами) между brain-emb
  и CLIP-image-emb, обучаемый `logit_scale = clamp(log_scale.exp(), max=100)`.
- **Регрессия:** `1 - cos(ĉ, c_clip)` + MSE — притягивает к конкретному эмбеддингу.
- **Diffusion prior:** простой MSE-объектив диффузии по CLIP-эмбеддингу (как в unCLIP prior).
- **(SSL-этап):** reconstruction-loss маскированных EEG-патчей / предсказание кодов
  нейротокенизатора.

---

## 5. Метрики (заменить «только FID»)

- Семантика: **CLIP-similarity**, **2-way / n-way top-k identification**.
- Низкоуровневая: SSIM, PixCorr, AlexNet(2,5)-точность.
- Высокоуровневая: Inception, CLIP, EfficientNet, SwAV-дистанции (как в MindEye/Brain-Diffuser).
- Retrieval (image↔brain) — forward/backward top-1.

---

## 6. План внедрения (поэтапно, без «большого взрыва»)

**Фаза A — препроцессинг и энкодер (быстрый выигрыш, без смены генератора)**
1. Нормальный EEG-препроцессинг (MNE: bandpass 0.5–45 Гц, notch 50 Гц, ICA, поканальный z-score).
2. Заменить энкодер на EEG Conformer; починить temperature → обучаемый `logit_scale`;
   добавить позиционное кодирование/CLS.
3. Заменить ResNet50 на замороженный CLIP ViT-L как цель выравнивания.

**Фаза B — самообучение**
4. Masked-EEG претрейн энкодера на всех доступных (в т.ч. неразмеченных) EEG, затем
   дообучение головы выравнивания на парах.

**Фаза C — генерация на диффузии**
5. Подключить замороженную SD (unCLIP/IP-Adapter), генерация из предсказанного CLIP-emb.
6. Добавить Diffusion Prior между энкодером и SD.
7. (Опц.) low-level поток + img2img для структуры.

**Фаза D — обобщение**
8. Subject-эмбеддинги, аугментации EEG, метрики идентификации/CLIP-sim, retrieval-оценка.

Каждая фаза самостоятельна и измерима: A и C дают наибольший прирост и их можно делать
независимо.

---

## 7. Эскизы ключевых новых модулей

> Псевдокод-ориентир для `models/`, не финальный код.

```python
# 1. EEG Conformer-энкодер (вместо LSTM/vanilla-Transformer)
class EEGConformer(nn.Module):
    def __init__(self, in_ch=128, emb=768, depth=6, heads=8, n_subjects=1):
        super().__init__()
        # пространственно-временной токенизатор
        self.tokenizer = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding=(0, 12)),   # временной
            nn.Conv2d(40, 40, (in_ch, 1)),                # пространственный (по каналам)
            nn.BatchNorm2d(40), nn.ELU(),
            nn.AvgPool2d((1, 15), (1, 5)), nn.Dropout(0.3),
        )
        self.proj = nn.LazyLinear(emb)
        self.pos = nn.Parameter(torch.randn(1, 1, emb) * 0.02)  # + позиционное кодирование
        self.cls = nn.Parameter(torch.randn(1, 1, emb) * 0.02)
        layer = nn.TransformerEncoderLayer(emb, heads, emb*4, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.subject = nn.Embedding(n_subjects, emb)

    def forward(self, x, subj_id=None):           # x: (B, C, T)
        h = self.tokenizer(x.unsqueeze(1))        # (B, 40, 1, T')
        h = h.flatten(2).transpose(1, 2)          # (B, T', 40)
        h = self.proj(h)
        if subj_id is not None:
            h = h + self.subject(subj_id).unsqueeze(1)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = self.encoder(torch.cat([cls, h], 1) + self.pos)
        return h[:, 0]                            # brain embedding (B, emb)


# 2. Diffusion Prior (brain emb -> распределение CLIP image emb)
class DiffusionPrior(nn.Module):
    """Обучается восстанавливать c_clip из зашумлённого таргета,
    обусловленного brain-эмбеддингом (как unCLIP prior)."""
    def __init__(self, dim=768, depth=6, heads=8):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, dim*4, batch_first=True)
        self.net = nn.TransformerEncoder(layer, depth)
        self.time_mlp = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, noisy_clip, t, brain_emb):
        tok = torch.stack([self.time_mlp(t[:, None]), brain_emb, noisy_clip], dim=1)
        return self.net(tok)[:, -1]               # предсказанный CLIP-emb
```

```python
# 3. Генерация: замороженная SD unCLIP из предсказанного CLIP-эмбеддинга
#    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(...)
#    image = pipe(image_embeds=predicted_clip_emb).images[0]
#    (или IP-Adapter: подать predicted_clip_emb в cross-attention SDXL)
```

---

## 8. Краткий вывод

EEG→image сегодня — самый реалистичный сценарий неинвазивного безопасного BCI, и
основной рычаг качества лежит **не в генераторе картинок**, а в:
1. честном препроцессинге EEG,
2. самообучении энкодера (Masked-EEG / foundation-модель),
3. выравнивании в готовое CLIP-пространство,
4. генерации замороженной диффузией с диффузионным приором.

StyleGAN2-ADA из 1.0 имеет смысл оставить только как лёгкий baseline; целевой путь —
CLIP-aligned энкодер + Diffusion Prior + Stable Diffusion. Это переносит на проект
весь зрительный приор больших моделей и снимает главный потолок EEG — нехватку данных.
