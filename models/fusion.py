# models/fusion.py
# ✅ VERSION FINALE CORRIGÉE

import torch
import torch.nn as nn
from .aasist import AASIST


class HybridFusionModel(nn.Module):
    """
    Modèle de Fusion Hybride : AASIST + LFCC.

    Deux branches parallèles :
      1. AASIST    : waveform brute → embedding (B, aasist_embed_dim=160)
      2. LFCC      : 60 coefficients → GRU → embedding (B, hidden_dim=128)

    Fusion :
      concat(160 + 128 = 288) → classifieur → logit binaire

    Args:
        lfcc_dim         : 60   (= config.lfcc.n_filters)
        aasist_embed_dim : 160  (= config.fusion.aasist_dim = config.aasist.embedding_dim)
        hidden_dim       : 128  (= config.fusion.hidden_dim)
        dropout          : 0.3
    """

    def __init__(self, lfcc_dim=60, aasist_embed_dim=160,
                 hidden_dim=128, dropout=0.3):
        super().__init__()

        # ── Branche 1 : AASIST ──────────────────────────────────
        self.aasist = AASIST(
            sinc_filters    = 64,
            graph_nodes     = 128,
            attention_heads = 4,
            embedding_dim   = aasist_embed_dim,  # 160
        )

        # ── Branche 2 : LFCC ────────────────────────────────────
        self.lfcc_projection = nn.Sequential(
            nn.Linear(lfcc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lfcc_temporal = nn.GRU(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            batch_first = True,
        )

        # ── Classifieur de fusion ────────────────────────────────
        # Entrée = aasist_embed_dim + hidden_dim = 160 + 128 = 288
        fusion_dim = aasist_embed_dim + hidden_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Logit binaire
        )

    def forward(self, waveform: torch.Tensor,
                lfcc_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform      : (B, 1, T)
            lfcc_features : (B, T', 60)

        Returns:
            logits : (B, 1)
                     positif → authentique / négatif → deepfake
        """
        # Branche AASIST
        aasist_embed = self.aasist(waveform)              # (B, 160)

        # Branche LFCC
        lfcc_proj            = self.lfcc_projection(lfcc_features)  # (B, T', 128)
        lfcc_seq, _          = self.lfcc_temporal(lfcc_proj)        # (B, T', 128)
        lfcc_embed           = torch.mean(lfcc_seq, dim=1)          # (B, 128)

        # Fusion
        combined = torch.cat([aasist_embed, lfcc_embed], dim=1)     # (B, 288)
        logits   = self.fusion(combined)                             # (B, 1)

        return logits