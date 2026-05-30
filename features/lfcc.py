# features/lfcc.py

import torch
import numpy as np


class LFCCExtractor:
    """
    LFCC — Linear Frequency Cepstral Coefficients.

    Filtres espacés LINÉAIREMENT (vs Mel pour MFCC)
    → meilleure sensibilité aux artefacts HF des systèmes TTS.

    Args:
        sample_rate : Hz (défaut: 16000)
        n_filters   : Filtres triangulaires linéaires (défaut: 60)
        n_ceps      : Coefficients cepstraux gardés (défaut: 60)
        window_ms   : Fenêtre d'analyse en ms (défaut: 25)
        hop_ms      : Pas entre fenêtres en ms (défaut: 10)
    """

    def __init__(self, sample_rate=16000, n_filters=60, n_ceps=60,
                 window_ms=25, hop_ms=10):
        self.sample_rate = sample_rate
        self.n_filters   = n_filters
        self.n_ceps      = n_ceps
        self.win_length  = int(window_ms * sample_rate / 1000)
        self.hop_length  = int(hop_ms    * sample_rate / 1000)
        self.n_fft       = self.win_length

        # Banc de filtres construit une seule fois
        self.filters = self._create_filters()  # (n_filters, n_fft//2+1)

    def _create_filters(self) -> torch.Tensor:
        """Filtres triangulaires linéairement espacés."""
        n_bins    = self.n_fft // 2 + 1
        f_max     = self.sample_rate / 2.0
        centers   = np.linspace(0, f_max, self.n_filters + 2)
        fft_freqs = np.linspace(0, f_max, n_bins)

        filters = np.zeros((self.n_filters, n_bins), dtype=np.float32)
        for i in range(self.n_filters):
            left   = centers[i]
            center = centers[i + 1]
            right  = centers[i + 2]
            rising  = (fft_freqs - left)  / (center - left  + 1e-8)
            falling = (right - fft_freqs) / (right  - center + 1e-8)
            filters[i] = np.maximum(0.0, np.minimum(rising, falling))

        return torch.from_numpy(filters)  # (n_filters, n_bins)

    def _dct(self, x: torch.Tensor) -> torch.Tensor:
        """
        DCT de type II.
        Args:
            x : (n_filters, n_frames)
        Returns:
            (n_ceps, n_frames)
        """
        N      = x.shape[0]
        device = x.device

        k = torch.arange(self.n_ceps, device=device).float().unsqueeze(1)  # (n_ceps, 1)
        n = torch.arange(N,           device=device).float().unsqueeze(0)  # (1, N)

        # Formule DCT-II : cos(π/N * k * (n + 0.5))
        dct_matrix     = torch.cos(np.pi / N * k * (n + 0.5))  # (n_ceps, N)
        dct_matrix[0]  *= np.sqrt(1.0 / N)
        dct_matrix[1:] *= np.sqrt(2.0 / N)

        return torch.mm(dct_matrix, x)  # (n_ceps, n_frames)

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extrait les LFCC d'un signal audio.

        Returns:
            lfcc : (n_frames, n_ceps)
        """
        device = waveform.device

        # Uniformiser en 1D pour torch.stft
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        # STFT
        stft = torch.stft(
            waveform,
            n_fft       = self.n_fft,
            hop_length  = self.hop_length,
            win_length  = self.win_length,
            window      = torch.hann_window(self.win_length, device=device),
            return_complex = True,
        )
        # (n_fft//2+1, n_frames)

        power_spec    = torch.abs(stft) ** 2
        filters       = self.filters.to(device)
        filter_energy = torch.mm(filters, power_spec)   # (n_filters, n_frames)
        log_energy    = torch.log(filter_energy + 1e-10)
        lfcc          = self._dct(log_energy)            # (n_ceps, n_frames)

        return lfcc.T  # (n_frames, n_ceps)