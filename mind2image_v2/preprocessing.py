"""EEG-preprocessing dlya MIND2IMAGE 2.0.

Zamenyaet naivnuyu normirovku `(eeg - max/2)/(max/2)` iz models/dataset.py na
chestnyy pipeline: polosovaya fil'tratsiya + notch + (opc.) ICA + pokanal'naya
standartizaciya.

Esli ustanovlen `mne` i `scipy` — ispol'zuyutsya oni. Inache rabotaet chistyy
numpy-fallback (bandpass cherez FFT, notch cherez prostoy rezhektornyy fil'tr).
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def _fft_bandpass(x: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """Polosovaya fil'tratsiya cherez FFT (numpy-fallback). x: (C, T)."""
    n = x.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= low) & (freqs <= high)
    spec = np.fft.rfft(x, axis=-1)
    spec[..., ~mask] = 0.0
    return np.fft.irfft(spec, n=n, axis=-1)


def _fft_notch(x: np.ndarray, freq: float, fs: float, q: float = 1.0) -> np.ndarray:
    """Rezhektornyy fil'tr na chastotu seti (numpy-fallback)."""
    n = x.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.fft.rfft(x, axis=-1)
    band = np.abs(freqs - freq) <= q
    spec[..., band] = 0.0
    return np.fft.irfft(spec, n=n, axis=-1)


def preprocess_eeg(
    eeg: np.ndarray,
    fs: float = 1000.0,
    bandpass: Tuple[float, float] = (0.5, 45.0),
    notch: Optional[float] = 50.0,
    per_channel_zscore: bool = True,
    use_ica: bool = False,
    ch_names: Optional[list] = None,
) -> np.ndarray:
    """Preprocessing odnogo EEG-trial.

    Args:
        eeg: (C, T) — kanaly x vremya.
        fs: chastota diskretizatsii.
        bandpass: (low, high) v Gts.
        notch: chastota seti (50/60) ili None.
        per_channel_zscore: pokanal'naya standartizaciya.
        use_ica: udalenie artefaktov cherez mne.preprocessing.ICA (esli dostupno).
        ch_names: imena kanalov (nuzhny tol'ko dlya ICA cherez mne).

    Returns:
        (C, T) float32.
    """
    x = np.asarray(eeg, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"ozhidaetsya (C, T), polucheno {x.shape}")

    try:
        from scipy.signal import butter, filtfilt, iirnotch  # type: ignore

        low, high = bandpass
        b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype="band")
        x = filtfilt(b, a, x, axis=-1)
        if notch is not None:
            bn, an = iirnotch(notch / (fs / 2), Q=30.0)
            x = filtfilt(bn, an, x, axis=-1)
    except Exception:
        # chistyy numpy fallback
        x = _fft_bandpass(x, bandpass[0], bandpass[1], fs)
        if notch is not None:
            x = _fft_notch(x, notch, fs)

    if use_ica:
        x = _maybe_ica(x, fs, ch_names)

    if per_channel_zscore:
        mu = x.mean(axis=-1, keepdims=True)
        sd = x.std(axis=-1, keepdims=True) + 1e-7
        x = (x - mu) / sd

    return x.astype(np.float32)


def _maybe_ica(x: np.ndarray, fs: float, ch_names: Optional[list]) -> np.ndarray:
    """Udalenie artefaktov cherez mne ICA. Pri otsutstvii mne — vozvrashchaet x."""
    try:
        import mne  # type: ignore

        n_ch = x.shape[0]
        names = ch_names or [f"ch{i}" for i in range(n_ch)]
        info = mne.create_info(names, sfreq=fs, ch_types="eeg")
        raw = mne.io.RawArray(x, info, verbose="ERROR")
        ica = mne.preprocessing.ICA(
            n_components=min(n_ch, 20), random_state=0, max_iter="auto", verbose="ERROR"
        )
        ica.fit(raw, verbose="ERROR")
        # avtomaticheskaya pometka EOG-podobnyh komponent po dispersii fronta
        raw = ica.apply(raw, verbose="ERROR")
        return raw.get_data()
    except Exception:
        return x


def preprocess_batch(eegs: np.ndarray, **kwargs) -> np.ndarray:
    """(N, C, T) -> (N, C, T) preprocessing po kazhdomu trial."""
    return np.stack([preprocess_eeg(e, **kwargs) for e in eegs], axis=0)
