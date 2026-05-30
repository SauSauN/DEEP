# models/aasist.py
# ✅ VERSION FINALE CORRIGÉE
#
# SOLUTION RETENUE pour le bug GRU bidirectionnel :
#   GRU bidirectionnel conservé (hidden=64 → sortie=128)
#   HSGAL node_dim = sinc_filters * 2 = 128  ← aligné
#   projection : Linear(128 → embedding_dim)
#
# Cohérence complète :
#   sinc_filters=64 → GRU sortie=128 → HSGAL(128) → projection(128→160)
#   → embedding_dim=160 → config.fusion.aasist_dim=160 ✅

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SincConv(nn.Module):
    """
    Filtres sinc paramétrables appris sur l'audio brut.

    Args:
        in_channels  : 1 (mono)
        out_channels : Nombre de filtres
        kernel_size  : Longueur du noyau
        stride, padding : Paramètres convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding

        self.low_hz  = nn.Parameter(torch.randn(out_channels, 1))
        self.high_hz = nn.Parameter(torch.randn(out_channels, 1))
        self._init_frequencies()

        # Fenêtre de Hamming fixe enregistrée comme buffer
        self.register_buffer('window', torch.hamming_window(kernel_size))

    def _init_frequencies(self):
        """Initialisation avec fréquences Mel pour bonne couverture."""
        mel_max   = 2595 * np.log10(1 + 8000 / 700)
        mel_freqs = torch.linspace(0, mel_max, self.out_channels)
        hz_freqs  = 700 * (10 ** (mel_freqs / 2595) - 1)
        with torch.no_grad():
            self.low_hz.data  = hz_freqs.view(-1, 1) / 8000
            self.high_hz.data = (hz_freqs + 200).view(-1, 1) / 8000

    def _sinc(self, x):
        return torch.where(x == 0, torch.ones_like(x),
                           torch.sin(x) / (x + 1e-8))

    def forward(self, x):
        """
        Args:
            x : (B, 1, T)
        Returns:
            (B, out_channels, T)
        """
        low  = self.low_hz  * 8000
        high = self.high_hz * 8000
        band = high - low

        t = torch.linspace(0, 1, self.kernel_size, device=x.device) - 0.5
        t = t.unsqueeze(0).unsqueeze(0) * np.pi * 2 * band.unsqueeze(-1)

        band_pass = (2 * band.unsqueeze(-1) * self._sinc(t)
                     * torch.cos(2 * np.pi * low.unsqueeze(-1) * t))
        band_pass = band_pass * self.window.to(x.device)

        return F.conv1d(x, band_pass, stride=self.stride, padding=self.padding)


class RawNet2Encoder(nn.Module):
    """
    Encodeur RawNet2 : SincConv + blocs résiduels + GRU bidirectionnel.

    GRU bidirectionnel conservé :
      entrée = hidden_dim = 64
      sortie = hidden_dim * 2 = 128   ← HSGAL doit recevoir 128
    """

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=6):
        super().__init__()
        self.sinc_conv = SincConv(input_dim, hidden_dim,
                                  kernel_size=251, stride=1, padding=125)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm1d(hidden_dim),
            )
            for _ in range(num_layers)
        ])

        # GRU bidirectionnel → sortie = hidden_dim * 2 = 128
        self.gru = nn.GRU(hidden_dim, hidden_dim,
                          batch_first=True, bidirectional=True)

    def forward(self, x):
        """
        Args:
            x : (B, 1, T)
        Returns:
            x : (B, T, hidden_dim*2=128)
        """
        x = F.relu(self.bn1(self.sinc_conv(x)))  # (B, 64, T)

        for block in self.res_blocks:
            x = F.relu(block(x) + x)              # connexion résiduelle

        x = x.transpose(1, 2)                     # (B, T, 64)
        x, _ = self.gru(x)                        # (B, T, 128)
        return x


class HSGAL(nn.Module):
    """
    Graph Attention Layer (approximation via MultiheadAttention).

    node_dim = sinc_filters * 2 = 128  ← aligné avec sortie GRU
    """

    def __init__(self, node_dim=128, n_heads=4):
        super().__init__()
        assert node_dim % n_heads == 0, \
            f"node_dim ({node_dim}) doit être divisible par n_heads ({n_heads})"

        self.attn    = nn.MultiheadAttention(node_dim, n_heads,
                                              dropout=0.1, batch_first=True)
        self.norm    = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args/Returns:
            x : (B, T, node_dim=128)
        """
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))


class AASIST(nn.Module):
    """
    AASIST complet et fonctionnel.

    Chaîne de dimensions :
        (B,1,T) → SincConv(64) → GRU_bidir → (B,T,128)
        → HSGAL(128) × 2 → MeanPool → (B,128)
        → Linear(128→embedding_dim=160) → (B,160)

    Args:
        sinc_filters    : 64
        graph_nodes     : 128  (= sinc_filters * 2)
        attention_heads : 4
        embedding_dim   : 160  (= config.fusion.aasist_dim)
    """

    def __init__(self, sinc_filters=64, graph_nodes=128,
                 attention_heads=4, embedding_dim=160):
        super().__init__()

        self.encoder = RawNet2Encoder(
            input_dim=1, hidden_dim=sinc_filters
        )

        # node_dim = sinc_filters * 2 = 128 ← aligné avec GRU bidirectionnel
        self.graph_layers = nn.ModuleList([
            HSGAL(node_dim=sinc_filters * 2, n_heads=attention_heads)
            for _ in range(2)
        ])

        # sinc_filters * 2 → embedding_dim  (128 → 160)
        self.projection = nn.Sequential(
            nn.Linear(sinc_filters * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform : (B, 1, T)
        Returns:
            embedding : (B, embedding_dim=160)
        """
        x = self.encoder(waveform)          # (B, T, 128)

        for layer in self.graph_layers:
            x = layer(x)                    # (B, T, 128)

        x = torch.mean(x, dim=1)            # (B, 128)
        x = self.projection(x)              # (B, 160)
        return x