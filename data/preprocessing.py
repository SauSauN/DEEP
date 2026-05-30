# data/preprocessing.py

import torch
import torchaudio
import numpy as np


class AudioPreprocessor:
    """
    Standardise tous les fichiers audio pour le pipeline de détection.

    Pipeline :
      1. load_audio      — Chargement .flac/.wav + mono + rééchantillonnage
      2. normalize_volume — Normalisation [-1, +1]
      3. trim_silence    — Suppression des silences
      4. fix_length      — Durée fixe (4 secondes)
    """

    def __init__(self, target_length_sec: int = 4, sample_rate: int = 16000):
        self.sample_rate   = sample_rate
        self.target_length = target_length_sec * sample_rate

    def load_audio(self, file_path: str) -> torch.Tensor:
        """Charge .flac ou .wav → tensor (1, n_samples)."""
        waveform, sr = torchaudio.load(file_path)

        # Stéréo → mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Rééchantillonnage si nécessaire
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform  = resampler(waveform)

        return waveform  # (1, n_samples)

    def normalize_volume(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalisation par le pic → valeurs entre -1 et +1."""
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        return waveform

    def trim_silence(self, waveform: torch.Tensor,
                     threshold_db: float = -40.0) -> torch.Tensor:
        """
        Supprime les silences en début et fin."""
        threshold_amp = 10 ** (threshold_db / 20.0)
        signal        = waveform.squeeze(0)          # (T,)
        energy        = torch.abs(signal)
        non_silent    = torch.where(energy > threshold_amp)[0]

        if len(non_silent) == 0:
            return waveform  # Tout silence → retourner tel quel

        margin = int(0.1 * self.sample_rate)  # 100ms de marge
        start  = max(0, int(non_silent[0].item())  - margin)
        end    = min(signal.shape[0], int(non_silent[-1].item()) + margin)

        return waveform[:, start:end]  # (1, n_trimmed)

    def fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Ajuste à target_length."""
        current = waveform.shape[1]

        if current < self.target_length:
            repeats  = int(np.ceil(self.target_length / current))
            waveform = waveform.repeat(1, repeats)
            waveform = waveform[:, :self.target_length]

        elif current > self.target_length:
            waveform = waveform[:, :self.target_length]

        return waveform  # (1, target_length)

    def process(self, file_path: str) -> torch.Tensor:
        """Pipeline complet → tensor (1, target_length)."""
        waveform = self.load_audio(file_path)
        waveform = self.normalize_volume(waveform)
        waveform = self.trim_silence(waveform)
        waveform = self.fix_length(waveform)
        waveform = self.normalize_volume(waveform)  # Re-normalisation finale
        return waveform