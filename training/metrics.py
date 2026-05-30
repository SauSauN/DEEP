# training/metrics.py

import numpy as np
from sklearn.metrics import roc_curve


def calculate_eer(scores, labels):
    """
    Equal Error Rate (EER).

    Seuil θ* où FAR == FRR.
    Plus bas = meilleur. Objectif bancaire : < 2%.

    Args:
        scores : array [0,1] (1 = plus authentique)
        labels : array (1=réel, 0=spoof)

    Returns:
        (eer_percent, optimal_threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr      = 1.0 - tpr
    eer_idx  = np.nanargmin(np.abs(fnr - fpr))
    eer      = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0 * 100.0)
    threshold = float(thresholds[eer_idx])
    return eer, threshold


def calculate_tdcf(scores, labels, p_target=0.01, c_miss=10, c_fa=1):
    """
    tandem Detection Cost Function (t-DCF).

    Contexte bancaire : c_miss >> c_fa
    (fraude non détectée coûte plus que client rejeté).

    Returns:
        min_dcf : plus bas = mieux
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr     = 1.0 - tpr
    dcf     = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    return float(np.min(dcf))