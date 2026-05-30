# inference/detect.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import argparse
from pathlib import Path

from data.preprocessing import AudioPreprocessor
from features.lfcc      import LFCCExtractor
from models.fusion      import HybridFusionModel


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Charge le modele entraine depuis un checkpoint."""
    model = HybridFusionModel(
        lfcc_dim         = config['fusion']['lfcc_dim'],
        # CORRECTION : 'aasist_dim' (160) et non 'graph_nodes' (128)
        aasist_embed_dim = config['fusion']['aasist_dim'],
        hidden_dim       = config['fusion']['hidden_dim'],
        dropout          = config['fusion']['dropout'],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    eer       = checkpoint.get('eer', None)
    threshold = checkpoint.get('threshold', 0.5)

    return model, eer, threshold


def detect_audio(file_path: str, model, preprocessor,
                 lfcc_extractor, device: torch.device) -> float:
    """
    Analyse un fichier audio et retourne le score de liveness.

    Returns:
        score : float [0, 1]
                proche de 1 -> voix authentique
                proche de 0 -> deepfake detecte
    """
    # Pretraitement : (1, 64000)
    waveform = preprocessor.process(file_path)

    # Ajout dimension batch : (1, 1, 64000)
    waveform_batch = waveform.unsqueeze(0).to(device)

    # Extraction LFCC depuis waveform (1, T)
    # lfcc_extractor.extract accepte (1, T) ou (T,)
    lfcc       = lfcc_extractor.extract(waveform)   # (T', 60)
    lfcc_batch = lfcc.unsqueeze(0).to(device)       # (1, T', 60)

    with torch.no_grad():
        logits = model(waveform_batch, lfcc_batch)  # (1, 1)
        score  = torch.sigmoid(logits).item()       # float [0, 1]

    return score


def main():
    parser = argparse.ArgumentParser(description='VoiceGuard - Detection de deepfakes vocaux')
    parser.add_argument('audio_file',    type=str,   help='Chemin vers le fichier audio a analyser')
    parser.add_argument('--checkpoint',  type=str,   default='checkpoints/best_model.pth')
    parser.add_argument('--config',      type=str,   default='config.yaml')
    parser.add_argument('--threshold',   type=float, default=None,
                        help='Seuil de decision (defaut: valeur optimale du checkpoint)')
    args = parser.parse_args()

    # Verification que le fichier audio existe
    if not os.path.exists(args.audio_file):
        print(f"ERREUR : fichier audio introuvable : {args.audio_file}")
        sys.exit(1)

    # Verification que le checkpoint existe
    if not os.path.exists(args.checkpoint):
        print(f"ERREUR : checkpoint introuvable : {args.checkpoint}")
        print(f"Lancez d'abord l'entrainement : python training/train.py")
        sys.exit(1)

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Device
    device_str = config['training']['device']
    device     = torch.device(
        device_str if device_str == 'cpu' or torch.cuda.is_available() else 'cpu'
    )

    # Processeurs audio
    preprocessor = AudioPreprocessor(
        target_length_sec = config['audio']['target_length_seconds'],
        sample_rate       = config['audio']['sample_rate'],
    )

    lfcc_extractor = LFCCExtractor(
        sample_rate = config['audio']['sample_rate'],
        n_filters   = config['lfcc']['n_filters'],
        n_ceps      = config['lfcc']['n_filters'],
        window_ms   = config['lfcc']['window_ms'],
        hop_ms      = config['lfcc']['hop_ms'],
    )

    # Chargement du modele (avec le bon aasist_embed_dim=160)
    model, eer, checkpoint_threshold = load_model(args.checkpoint, config, device)

    # Seuil de decision : priorite a l'argument CLI, sinon checkpoint, sinon 0.5
    threshold = args.threshold if args.threshold is not None else checkpoint_threshold

    print(f"\nModele charge")
    if eer is not None:
        print(f"  EER entrainement : {eer:.2f}%")
    print(f"  Seuil de decision : {threshold:.4f}")
    print(f"  Device : {device}")

    # Detection
    score   = detect_audio(args.audio_file, model, preprocessor, lfcc_extractor, device)
    is_real = score >= threshold

    # Affichage du verdict
    print("\n" + "=" * 55)
    print(f"  Fichier : {args.audio_file}")
    print(f"  Score   : {score:.4f}  (0=fake, 1=reel)")
    print(f"  Seuil   : {threshold:.4f}")
    print(f"  Verdict : {'AUTHENTIQUE (Bonafide)' if is_real else 'DEEPFAKE DETECTE'}")
    print("=" * 55)

    return 0 if is_real else 1


if __name__ == '__main__':
    sys.exit(main())