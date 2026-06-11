"""Dataset dlya MIND2IMAGE 2.0.

V otlichie ot models/dataset.py (1.0), kotoryy srazu prognyal EEG cherez encoder i
hranil tol'ko fichi, zdes' hranyatsya syrye (predobrabotannye) EEG + kartinka +
subject_id. Encoder uchitsya end-to-end, a CLIP-targety schitayutsya na letu.

Format faylov sovmestim so staroy versiey: np.load(path) -> [image, eeg(T,C), label].
"""

from __future__ import annotations

from typing import Optional, Tuple, List
from glob import glob
import numpy as np

import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_eeg


class EEGImageDatasetV2(Dataset):
    """Vozvrashchaet (eeg[C,T], image[3,H,W], label, subject_id)."""

    def __init__(
        self,
        dataset_glob: str,
        image_size: Tuple[int, int] = (224, 224),
        fs: float = 1000.0,
        bandpass: Tuple[float, float] = (0.5, 45.0),
        notch: Optional[float] = 50.0,
        use_ica: bool = False,
        subject_ids: Optional[List[int]] = None,
    ):
        import cv2  # lokal'nyy import, chtoby paket importirovalsya bez opencv

        paths = sorted(glob(dataset_glob))
        if not paths:
            raise FileNotFoundError(f"net faylov po patternu: {dataset_glob}")

        self.eegs, self.images, self.labels, self.subjects = [], [], [], []
        for i, path in enumerate(paths):
            arr = np.load(path, allow_pickle=True)
            eeg = np.float32(arr[1].T)                      # (C, T)
            eeg = preprocess_eeg(eeg, fs=fs, bandpass=bandpass,
                                 notch=notch, use_ica=use_ica)
            img = cv2.resize(arr[0], image_size)
            img = np.float32(np.transpose(img, (2, 0, 1))) / 255.0
            self.eegs.append(eeg)
            self.images.append(img)
            self.labels.append(int(arr[2]))
            self.subjects.append(subject_ids[i] if subject_ids else 0)

        self.eegs = torch.from_numpy(np.stack(self.eegs)).float()
        self.images = torch.from_numpy(np.stack(self.images)).float()
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.subjects = torch.tensor(self.subjects, dtype=torch.long)

    def __len__(self) -> int:
        return self.eegs.shape[0]

    def __getitem__(self, idx: int):
        return self.eegs[idx], self.images[idx], self.labels[idx], self.subjects[idx]


class EEGOnlyDataset(Dataset):
    """Tol'ko EEG — dlya self-supervised pretrain (masked_pretrain)."""

    def __init__(self, dataset_glob: str, fs: float = 1000.0,
                 bandpass: Tuple[float, float] = (0.5, 45.0),
                 notch: Optional[float] = 50.0):
        paths = sorted(glob(dataset_glob))
        if not paths:
            raise FileNotFoundError(f"net faylov po patternu: {dataset_glob}")
        eegs = []
        for path in paths:
            arr = np.load(path, allow_pickle=True)
            eeg = np.float32(arr[1].T)
            eegs.append(preprocess_eeg(eeg, fs=fs, bandpass=bandpass, notch=notch))
        self.eegs = torch.from_numpy(np.stack(eegs)).float()

    def __len__(self) -> int:
        return self.eegs.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.eegs[idx]
